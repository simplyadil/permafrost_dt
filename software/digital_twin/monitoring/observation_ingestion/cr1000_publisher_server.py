"""CR1000 datalogger to RabbitMQ sensor publisher.

This is the *input gate* of the digital twin: it polls a CR1000 over RS232,
turns each newly-logged record into a ``sensor_message_2d`` snapshot and
publishes it to the ``permafrost.sensors.2d`` fanout exchange so every
downstream service (ingestion, boundary-condition builder, thaw-front
reconstruction, FEM, inversion) receives the same live feed.

The fetch engine is the batched/atomic/gap-free PakBus collector validated in
``software/tools/cr1000_monitor.py``. Key properties:

* **Batched collect** — the table is too wide for one PakBus response, so each
  poll issues several field-subset commands (with retries).
* **Atomic merge** — a record is only emitted once *every* batch has populated
  *all* its fields. A batch delayed by a retry can otherwise return a newer
  window than the others, producing a partially-filled record; such records are
  deferred to the next poll instead of being published half-empty.
* **Gap-free** — each poll asks for the last ``records_per_poll`` records and
  publishes every one not seen yet, so a slow poll backfills skipped records.
* **Reconnect loop** — any serial failure tears down the device and reconnects.
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pycampbellcr1000 import CR1000
    from pycampbellcr1000.compat import is_py3
    _CR1000_AVAILABLE = True
except ImportError:
    _CR1000_AVAILABLE = False
    CR1000 = None  # type: ignore[assignment,misc]
    is_py3 = True  # compat.is_py3 is a bool constant, not a callable

from software.digital_twin.communication.messaging import RabbitMQClient, RabbitMQConfig
from software.digital_twin.data_access.influx_utils import InfluxConfig, InfluxHelper
from software.utils.logging_setup import get_logger

SENSOR_EXCHANGE = "permafrost.sensors.2d"
FLUID_EXCHANGE = "permafrost.fluid"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "sensor_message_2d.json"
FLUID_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "communication" / "schemas" / "fluid_message.json"

# PakBus collect tuning (mirrors the validated cr1000_monitor.py tool).
_MAX_RESP_PAYLOAD = 900     # bytes the logger packs into one collect response
_MAX_CMD_FIELDS = 60        # max field indices per collect command
_BATCH_MAX_RETRIES = 3
_BATCH_RETRY_DELAY = 2.0


def _decode_field_name(name) -> str:
    """Decode a bytes field name to str, stripping the b'' wrapper."""
    if isinstance(name, (bytes, bytearray)):
        return name.decode("utf-8", errors="replace")
    return str(name)


def _call_with_timeout(fn, timeout, *args, **kwargs):
    """Run ``fn`` in a daemon thread, raising TimeoutError if it overruns.

    Guards against a serial read hanging forever and stalling the service."""
    result = [None]
    error: List[Optional[BaseException]] = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller below
            error[0] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"call timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]


class CR1000PublisherServer:
    """Polls a CR1000 datalogger over RS232 and publishes sensor snapshots
    to the ``permafrost.sensors.2d`` fanout exchange so all downstream services
    (ingestion, thaw-front reconstruction, FEM, etc.) receive live data.

    ``sensor_map`` maps CR1000 field names to physical positions required by the
    ``sensor_message_2d`` schema. Field names must match the names the logger
    reports, including the array-index suffix (every Table1 column has BegIdx 1,
    so e.g. the ``H10D0`` column is referenced as ``"H10D0(1)"``)::

        sensor_map = [
            {"field": "H10D0(1)",  "sensor_id": "s1", "x_m": 0.10, "z_m": 0.00},
            {"field": "H10D30(1)", "sensor_id": "s2", "x_m": 0.10, "z_m": 0.30},
        ]
    """

    def __init__(
        self,
        port: str,
        baud_rate: int,
        table_name: str,
        sensor_map: List[Dict],
        rabbitmq_config: Optional[RabbitMQConfig] = None,
        *,
        pakbus_addr: int = 1,
        poll_interval_s: float = 30.0,
        site_id: str = "default",
        fluid_fields: Optional[Dict[str, str]] = None,
        fluid_exchange: Optional[str] = None,
        influx_config: Optional[InfluxConfig] = None,
        records_per_poll: int = 3,
        connect_timeout_s: float = 60.0,
        call_timeout_s: float = 120.0,
        reconnect_delay_s: float = 15.0,
    ) -> None:
        if not _CR1000_AVAILABLE:
            raise ImportError(
                "pycampbellcr1000 is not installed. Run: pip install pycampbellcr1000"
            )

        self.logger = get_logger("CR1000PublisherServer")
        self.port = port
        self.baud_rate = baud_rate
        self.table_name = table_name
        self.sensor_map = sensor_map
        self.pakbus_addr = pakbus_addr
        self.poll_interval_s = poll_interval_s
        self.site_id = site_id
        self.records_per_poll = max(1, int(records_per_poll))
        self.connect_timeout_s = connect_timeout_s
        self.call_timeout_s = call_timeout_s
        self.reconnect_delay_s = reconnect_delay_s

        self.rabbitmq_config = rabbitmq_config or RabbitMQConfig(
            exchange=SENSOR_EXCHANGE,
            exchange_type="fanout",
            schema_path=SCHEMA_PATH,
        )

        self._fluid_fields = fluid_fields or {}
        self._fluid_exchange = fluid_exchange
        self._influx_config = influx_config
        self._influx: Optional[InfluxHelper] = None

        self._device: Optional[CR1000] = None
        self._mq_client: Optional[RabbitMQClient] = None
        self._fluid_mq_client: Optional[RabbitMQClient] = None
        self._stop_event = threading.Event()
        self._epoch: Optional[datetime] = None

        # Dedup / progress tracking (persist across reconnects within a run).
        self._last_rec_nbr: Optional[int] = None
        self._last_datetime: Optional[datetime] = None

        # Per-connection table layout, rebuilt on each (re)connect.
        self._tabledef = None
        self._tablenbr: Optional[int] = None
        self._tabledefsig = None
        self._batches: List[List[int]] = []
        self._field_count: int = 0

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Connect to RabbitMQ (and Influx if fluid logging is enabled).

        The CR1000 serial connection is (re)established inside the polling loop
        so transient hardware failures don't crash startup."""
        if self._mq_client is None:
            self._mq_client = RabbitMQClient(self.rabbitmq_config)
        if self._fluid_fields and self._fluid_exchange and self._fluid_mq_client is None:
            fluid_config = RabbitMQConfig(
                host=self.rabbitmq_config.host,
                queue="",
                exchange=self._fluid_exchange,
                exchange_type="fanout",
                schema_path=FLUID_SCHEMA_PATH,
                username=self.rabbitmq_config.username,
                password=self.rabbitmq_config.password,
            )
            self._fluid_mq_client = RabbitMQClient(fluid_config)
            self.logger.info("Fluid temperature publisher ready (exchange=%s)", self._fluid_exchange)
        if self._fluid_fields and self._influx_config is not None and self._influx is None:
            self._influx = InfluxHelper(self._influx_config)
            self.logger.info("InfluxDB client ready for fluid temperature writes")
        self.logger.info(
            "CR1000PublisherServer ready (port=%s, table=%s, exchange=%s, sensors=%d)",
            self.port, self.table_name, self.rabbitmq_config.exchange, len(self.sensor_map),
        )

    def start(self) -> None:
        """Run the connect/poll/reconnect loop (blocking)."""
        if self._mq_client is None:
            self.setup()

        if not self.sensor_map:
            self.logger.warning(
                "sensor_map is empty — no field-to-position mappings defined, "
                "so no sensor snapshots will be published."
            )

        self._stop_event.clear()
        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.info("Interrupt received; shutting down")
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._stop_event.set()

    def close(self) -> None:
        """Stop polling and release all connections."""
        self.stop()
        self._disconnect_cr1000()
        if self._mq_client is not None:
            self._mq_client.disconnect()
            self._mq_client = None
        if self._fluid_mq_client is not None:
            self._fluid_mq_client.disconnect()
            self._fluid_mq_client = None
        if self._influx is not None:
            self._influx.close()
            self._influx = None
        self.logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # CONNECT / RECONNECT LOOP
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connect_cr1000()
                self._prepare_table()
                self._poll_loop()
            except Exception as exc:  # noqa: BLE001 - log and reconnect
                self.logger.error("CR1000 session error: %s", exc)
                self.logger.debug(traceback.format_exc())
            finally:
                self._disconnect_cr1000()

            if self._stop_event.is_set():
                break
            self.logger.info("Reconnecting to CR1000 in %.0fs...", self.reconnect_delay_s)
            self._stop_event.wait(timeout=self.reconnect_delay_s)

    def _connect_cr1000(self) -> None:
        address = f"serial:{self.port}:{self.baud_rate}"
        self.logger.info(
            "Connecting to CR1000 on %s at %d baud (PakBus addr=%d)",
            self.port, self.baud_rate, self.pakbus_addr,
        )
        t0 = time.time()
        try:
            self._device = _call_with_timeout(
                CR1000.from_url, self.connect_timeout_s, address, dest_addr=self.pakbus_addr
            )
        except Exception as exc:
            self.logger.error(
                "CR1000 connection failed: %s. "
                "Check: port correct, baud matches CR1000, PC400/LoggerNet closed "
                "(the serial port is exclusive), cable connected.",
                exc,
            )
            raise
        self.logger.info("CR1000 connection established in %.1fs", time.time() - t0)

    def _disconnect_cr1000(self) -> None:
        if self._device is not None:
            try:
                self._device.bye()
            except Exception:  # noqa: BLE001 - best effort
                pass
            # bye() only sends the PakBus goodbye packet — it does NOT release the
            # serial port. The link is otherwise closed only by PakBus.__del__ via
            # GC, which does not run promptly (the driver objects form reference
            # cycles), so the handle lingers open in THIS process and the next
            # reconnect fails with "Access is denied" on the same COM port.
            # Close the underlying link explicitly so the port is freed every time.
            try:
                link = getattr(getattr(self._device, "pakbus", None), "link", None)
                if link is not None:
                    link.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._device = None

    # ------------------------------------------------------------------
    # TABLE LAYOUT (rebuilt per connection)
    # ------------------------------------------------------------------

    def _prepare_table(self) -> None:
        """Fetch the table definition and pre-compute collect batches."""
        device = self._device
        self.logger.info("Fetching table definition (.TDF)...")
        raw = device.getfile('.TDF')
        self._tabledef = device.pakbus.parse_tabledef(raw)

        tname_enc = self.table_name.encode() if is_py3 else self.table_name
        tablenbr = None
        for i, item in enumerate(self._tabledef):
            if item['Header']['TableName'] == tname_enc:
                tablenbr = i + 1
                break
        if tablenbr is None:
            raise ValueError(f"Table '{self.table_name}' not found in TDF")

        t_def = self._tabledef[tablenbr - 1]
        fields = t_def['Fields']
        self._tablenbr = tablenbr
        self._tabledefsig = t_def['Signature']
        self._field_count = len(fields)

        DATATYPE = device.pakbus.DATATYPE
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_size = 8
        for i, fld in enumerate(fields, 1):
            ftype = fld['FieldType']
            dim = fld['Dimension']
            if ftype == 'ASCII':
                fsize = dim
            elif isinstance(ftype, str) and ftype in DATATYPE and DATATYPE[ftype]['size']:
                fsize = dim * DATATYPE[ftype]['size']
            else:
                fsize = dim * 4
            # Each command pulls records_per_poll records, so size for N copies.
            eff = fsize * self.records_per_poll
            if (current_size + eff > _MAX_RESP_PAYLOAD or len(current_batch) >= _MAX_CMD_FIELDS) \
                    and current_batch:
                batches.append(current_batch)
                current_batch, current_size = [], 8
            current_batch.append(i)
            current_size += eff
        if current_batch:
            batches.append(current_batch)
        self._batches = batches

        self.logger.info(
            "Table '%s' ready: %d fields, %d collect batch(es) (%s)",
            self.table_name, self._field_count, len(batches),
            " | ".join(str(len(b)) for b in batches),
        )

    # ------------------------------------------------------------------
    # FETCH (batched, atomic merge)
    # ------------------------------------------------------------------

    def _fetch_recent_records(self) -> List[dict]:
        """Return the last ``records_per_poll`` records, oldest-first.

        Each record is a dict with ``_RecNbr``, ``Datetime`` and one decoded
        key per sensor field. Batches are merged by record number so a record
        is only complete once every batch contributed."""
        device = self._device
        records: Dict[int, dict] = {}

        for bi, batch in enumerate(self._batches):
            cmd = device.pakbus.get_collectdata_cmd(
                self._tablenbr, self._tabledefsig,
                mode=0x05, p1=self.records_per_poll, p2=0, field_list=batch,
            )

            msg = None
            for attempt in range(1, _BATCH_MAX_RETRIES + 1):
                _hdr, msg, _send_time = device.send_wait(cmd)
                if msg:
                    break
                self.logger.warning(
                    "Batch %d: no response (attempt %d/%d) — retrying in %.0fs",
                    bi, attempt, _BATCH_MAX_RETRIES, _BATCH_RETRY_DELAY,
                )
                if attempt < _BATCH_MAX_RETRIES:
                    time.sleep(_BATCH_RETRY_DELAY)

            if not msg:
                raise RuntimeError(f"Batch {bi}: no response after {_BATCH_MAX_RETRIES} attempts")

            resp_code = msg.get('RespCode')
            if resp_code not in (None, 0):
                raise RuntimeError(f"Batch {bi}: RespCode=0x{resp_code:02x}")

            rec_data = msg.get('RecData')
            if not rec_data:
                self.logger.warning("Batch %d: empty RecData — skipping", bi)
                continue

            data, _ = device.pakbus.parse_collectdata(rec_data, self._tabledef, fieldnbr=batch)
            for rec in data:
                for item in rec.get('RecFrag', []):
                    if not isinstance(item, dict):
                        continue
                    rn = item.get('RecNbr')
                    if rn is None:
                        continue
                    slot = records.setdefault(rn, {})
                    slot['_RecNbr'] = rn
                    slot['Datetime'] = item.get('TimeOfRec')
                    for k, v in item.get('Fields', {}).items():
                        key = _decode_field_name(k) if isinstance(k, (bytes, bytearray)) else str(k)
                        slot[key] = v

        return [records[rn] for rn in sorted(records)]

    def _select_new(self, recs: List[dict]):
        """Filter to complete records newer than the last published one.

        Stops at the first incomplete record so a non-atomic batch never causes
        a record to be skipped — it's published complete on the next poll.
        On a cold start (no baseline yet) only the most recent record is taken,
        so the DT begins streaming from "now" rather than replaying the stored
        backlog."""
        if not recs:
            return [], None

        cold_start = self._last_rec_nbr is None and self._last_datetime is None
        pool = recs[-1:] if cold_start else recs

        new: List[dict] = []
        deferred = None
        for r in pool:
            rn = r['_RecNbr']
            dt = r.get('Datetime')
            if dt is None:
                continue
            if self._last_rec_nbr is not None and rn <= self._last_rec_nbr:
                continue
            if self._last_datetime is not None and dt <= self._last_datetime:
                continue
            nfields = len(r) - 2  # exclude _RecNbr + Datetime
            if nfields < self._field_count:
                deferred = (rn, nfields)
                break
            new.append(r)
        return new, deferred

    # ------------------------------------------------------------------
    # POLL LOOP
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.time()

            try:
                recs = _call_with_timeout(self._fetch_recent_records, self.call_timeout_s)
            except Exception as exc:  # noqa: BLE001 - reconnect
                self.logger.error("Fetch failed: %s — reconnecting", exc)
                self.logger.debug(traceback.format_exc())
                return  # back to _run() reconnect loop

            new, deferred = self._select_new(recs)

            if not recs:
                self.logger.warning("No record returned from CR1000")
            elif not new:
                if deferred:
                    self.logger.debug(
                        "Record #%s incomplete (%d/%d fields) — deferring to next poll",
                        deferred[0], deferred[1], self._field_count,
                    )
                else:
                    self.logger.debug("No new data (record #%s)", recs[-1]['_RecNbr'])
            else:
                published = 0
                for r in new:
                    if self._handle_record(r):
                        published += 1
                    self._last_rec_nbr = r['_RecNbr']
                    self._last_datetime = r['Datetime']
                tail = ("  (deferring #%s: %d/%d fields)"
                        % (deferred[0], deferred[1], self._field_count) if deferred else "")
                self.logger.info(
                    "Published %d snapshot(s)  |  latest #%s  |  %s  |  %.1fs fetch%s",
                    published, self._last_rec_nbr, self._last_datetime, time.time() - t0, tail,
                )

            elapsed = time.time() - t0
            self._stop_event.wait(timeout=max(0.0, self.poll_interval_s - elapsed))

    # ------------------------------------------------------------------
    # RECORD -> MESSAGE
    # ------------------------------------------------------------------

    def _handle_record(self, record: dict) -> bool:
        """Build and publish a sensor snapshot from one record. Returns True if
        a message was published."""
        message = self._build_message(record)
        if message is None:
            return False
        try:
            self._mq_client.publish(message)
        except Exception as exc:  # noqa: BLE001 - keep polling on broker hiccup
            self.logger.error("Failed to publish to RabbitMQ: %s", exc)
            return False
        self._emit_fluid_temps(record, message["time_hours"], message["timestamp"])
        return True

    def _compute_time_hours(self, record_dt: datetime) -> float:
        if self._epoch is None:
            self._epoch = record_dt
            self.logger.info("Time epoch set to %s", record_dt.isoformat())
        return (record_dt - self._epoch).total_seconds() / 3600.0

    def _build_message(self, record: dict) -> Optional[dict]:
        record_dt = record.get("Datetime")
        if record_dt is None:
            self.logger.warning("Record has no Datetime — skipping")
            return None

        time_hours = self._compute_time_hours(record_dt)

        sensors = []
        for entry in self.sensor_map:
            field = entry.get("field")
            if field is None:
                continue
            raw_val = record.get(field)
            if raw_val is None:
                self.logger.debug("Field '%s' not in record — skipping sensor", field)
                continue
            try:
                sensor: dict = {
                    "sensor_id": str(entry["sensor_id"]),
                    "x_m": float(entry["x_m"]),
                    "z_m": float(entry["z_m"]),
                    "temperature": float(raw_val),
                }
                if "y_m" in entry and entry["y_m"] is not None:
                    sensor["y_m"] = float(entry["y_m"])
            except (KeyError, TypeError, ValueError) as exc:
                self.logger.warning("Bad sensor_map entry for field '%s': %s", field, exc)
                continue
            sensors.append(sensor)

        if not sensors:
            self.logger.warning("No valid sensors built from record at t=%.4fh", time_hours)
            return None

        return {
            "time_hours": time_hours,
            "timestamp": record_dt.isoformat(),
            "sensors": sensors,
        }

    def _emit_fluid_temps(self, record: dict, time_hours: float, timestamp_iso: str) -> None:
        """Publish inlet/outlet fluid temps to the fluid fanout and/or Influx.

        Best-effort: a broker or Influx hiccup must never stop sensor polling."""
        if not self._fluid_fields:
            return
        inlet_field = self._fluid_fields.get("inlet")
        outlet_field = self._fluid_fields.get("outlet")
        if not inlet_field or not outlet_field:
            return
        try:
            t_in = float(record[inlet_field])
            t_out = float(record[outlet_field])
        except (KeyError, TypeError, ValueError) as exc:
            self.logger.debug("Could not read fluid temperature fields: %s", exc)
            return

        if self._fluid_mq_client is not None:
            try:
                self._fluid_mq_client.publish(
                    {
                        "time_hours": float(time_hours),
                        "timestamp": timestamp_iso,
                        "t_inlet_c": t_in,
                        "t_outlet_c": t_out,
                        "site_id": self.site_id,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep polling on broker hiccup
                self.logger.error("Failed to publish fluid temperatures to RabbitMQ: %s", exc)

        if self._influx is not None:
            try:
                self._influx.write_fluid_temperatures(
                    time_hours=time_hours,
                    t_inlet_c=t_in,
                    t_outlet_c=t_out,
                    site=self.site_id,
                )
            except Exception as exc:  # noqa: BLE001 - infra
                self.logger.error("Failed to write fluid temperatures to InfluxDB: %s", exc)
