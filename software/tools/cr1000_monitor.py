"""
CR1000 Monitoring Script
---------------------------------
Connects to a CR1000, polls for new records every POLL_INTERVAL seconds,
and appends each new record to a CSV file.

Requirements:
    pip install pycampbellcr1000
"""

import csv
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime

from pycampbellcr1000 import CR1000
from pycampbellcr1000.compat import is_py3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERIAL_PORT     = "COM5"
BAUD_RATE       = 115200
TABLE_NAME      = "Table1"
POLL_INTERVAL   = 30        # seconds between record checks
CONNECT_TIMEOUT = 60
CALL_TIMEOUT    = 120       # covers TDF retry worst-case
RECONNECT_DELAY = 15        # wait before reconnecting after a failure
CSV_FILE        = "cr1000_data.csv"
RECORDS_PER_POLL = 3        # fetch the last N records each poll so a slow
                            # poll backfills any skipped record (gap-free)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = "cr1000_monitor.log"

_fmt     = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S")
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file    = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file.setFormatter(_fmt)

log = logging.getLogger("cr1000_monitor")
log.setLevel(logging.DEBUG)
log.addHandler(_console)
log.addHandler(_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def call_with_timeout(fn, timeout, *args, **kwargs):
    result = [None]
    error  = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"call timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]


def _decode_field_name(name):
    """Decode bytes field name to str, stripping b'' wrapper."""
    if isinstance(name, (bytes, bytearray)):
        return name.decode("utf-8", errors="replace")
    return str(name)


# Only these sensors are used by the digital twin; everything else in Table1
# (PTemp_C, ref, VWC/EC/PA/VR moisture probes, etc.) is discarded.
_HXDY_RE    = re.compile(r"^H\d+D\d+$")          # thermistor strings, e.g. H10D0
_KEEP_EXACT = {"SHF", "SHF_2", "Inlet", "Outlet"}


def _keep_field(fname):
    """True if a Table1 field should be kept as a CSV column."""
    name = _decode_field_name(fname)
    return bool(_HXDY_RE.match(name)) or name in _KEEP_EXACT


def _decode_row(record):
    """Decode a fetched record's (possibly bytes) keys to str -> value."""
    row = {}
    for k, v in record.items():
        key = _decode_field_name(k) if isinstance(k, (bytes, bytearray)) else str(k)
        row[key] = v
    return row


# ---------------------------------------------------------------------------
# TDF — fetch and log structure, return tabledef
# ---------------------------------------------------------------------------
def fetch_tabledef(device):
    log.info("Fetching TDF...")
    raw      = device.getfile('.TDF')
    tabledef = device.pakbus.parse_tabledef(raw)
    log.info("TDF: %d bytes, tables: %s",
             len(raw), [_decode_field_name(t['Header']['TableName']) for t in tabledef])

    for t in tabledef:
        tname    = _decode_field_name(t['Header']['TableName'])
        interval = t['Header'].get('TblInterval', '?')
        log.info("  Table %r: %d fields, interval=%s", tname, len(t['Fields']), interval)
        for i, f in enumerate(t['Fields'], 1):
            log.debug("    [%3d] %-30s  type=%-8s  dim=%d  beg_idx=%s",
                      i, _decode_field_name(f['FieldName']), f['FieldType'],
                      f['Dimension'], f.get('BegIdx', 0))

    return tabledef


# ---------------------------------------------------------------------------
# Build ordered CSV column list from TDF (mirrors parse_collectdata key logic)
# ---------------------------------------------------------------------------
def build_columns(tabledef):
    tname_enc = TABLE_NAME.encode() if is_py3 else TABLE_NAME
    t_def = next((t for t in tabledef if t['Header']['TableName'] == tname_enc), None)
    if t_def is None:
        raise ValueError(f"Table '{TABLE_NAME}' not found in TDF")

    # Timestamp first, then one column per kept sensor label. RecNbr is tracked
    # internally for dedup but intentionally kept out of the CSV.
    columns = ['Timestamp']
    for fld in t_def['Fields']:
        fname   = fld['FieldName']
        if not _keep_field(fname):
            continue
        beg_idx = fld.get('BegIdx', 0)
        if beg_idx:
            if isinstance(fname, bytes):
                raw_key = fname + ('(%d)' % beg_idx).encode()
            else:
                raw_key = '%s(%d)' % (fname, beg_idx)
        else:
            raw_key = fname
        columns.append(_decode_field_name(raw_key))

    return columns


# ---------------------------------------------------------------------------
# Batched last-record fetch
# ---------------------------------------------------------------------------
def fetch_recent_records(device, tabledef):
    tname_enc = TABLE_NAME.encode() if is_py3 else TABLE_NAME
    tablenbr  = None
    for i, item in enumerate(tabledef):
        if item['Header']['TableName'] == tname_enc:
            tablenbr = i + 1
            break
    if tablenbr is None:
        raise ValueError(f"Table '{TABLE_NAME}' not found")

    t_def       = tabledef[tablenbr - 1]
    tabledefsig = t_def['Signature']
    fields      = t_def['Fields']

    DATATYPE         = device.pakbus.DATATYPE
    MAX_RESP_PAYLOAD = 900
    MAX_CMD_FIELDS   = 60

    batches, current_batch, current_size = [], [], 8

    for i, fld in enumerate(fields, 1):
        ftype = fld['FieldType']
        dim   = fld['Dimension']
        if ftype == 'ASCII':
            fsize = dim
        elif isinstance(ftype, str) and ftype in DATATYPE and DATATYPE[ftype]['size']:
            fsize = dim * DATATYPE[ftype]['size']
        else:
            fsize = dim * 4

        # We request RECORDS_PER_POLL records per command, so the response
        # carries N copies of each field — size the batch accordingly.
        eff = fsize * RECORDS_PER_POLL
        if (current_size + eff > MAX_RESP_PAYLOAD or len(current_batch) >= MAX_CMD_FIELDS) \
                and current_batch:
            batches.append(current_batch)
            current_batch, current_size = [], 8
        current_batch.append(i)
        current_size += eff

    if current_batch:
        batches.append(current_batch)

    log.debug("Batches: %d (%s fields each)",
              len(batches), " | ".join(str(len(b)) for b in batches))

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    records     = {}   # RecNbr -> {'_RecNbr', 'Timestamp', <sensor fields>}

    for bi, batch in enumerate(batches):
        cmd = device.pakbus.get_collectdata_cmd(
            tablenbr, tabledefsig, mode=0x05, p1=RECORDS_PER_POLL, p2=0, field_list=batch
        )

        msg = None
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.time()
            hdr, msg, send_time = device.send_wait(cmd)
            elapsed = time.time() - t0
            if msg:
                break
            log.warning("Batch %d: no response (attempt %d/%d, %.1fs) — retrying in %.0fs",
                        bi, attempt, MAX_RETRIES, elapsed, RETRY_DELAY)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        if not msg:
            raise RuntimeError(f"Batch {bi}: no response after {MAX_RETRIES} attempts")

        resp_code = msg.get('RespCode')
        if resp_code not in (None, 0):
            raise RuntimeError(f"Batch {bi}: RespCode=0x{resp_code:02x}")

        rec_data = msg.get('RecData')
        if not rec_data:
            log.warning("Batch %d: empty RecData — skipping", bi)
            continue

        data, _ = device.pakbus.parse_collectdata(rec_data, tabledef, fieldnbr=batch)

        for rec in data:
            for item in rec.get('RecFrag', []):
                if not isinstance(item, dict):
                    continue
                rn = item.get('RecNbr')
                if rn is None:
                    continue
                slot = records.setdefault(rn, {})
                slot['_RecNbr']   = rn
                slot['Timestamp'] = item.get('TimeOfRec')
                slot.update(item.get('Fields', {}))

    # Return records oldest-first so they are written in chronological order.
    return [records[rn] for rn in sorted(records)]


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------
def read_last_timestamp():
    """Return the Timestamp of the last row already in the CSV (or None).

    Used to seed dedup so restarting the script does not re-append records
    that were saved in a previous run."""
    if not (os.path.isfile(CSV_FILE) and os.path.getsize(CSV_FILE) > 0):
        return None
    try:
        with open(CSV_FILE, newline="", encoding="utf-8") as f:
            last = None
            for row in csv.DictReader(f):
                last = row.get('Timestamp')
        if not last:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(last, fmt)
            except ValueError:
                continue
        log.warning("Unrecognized timestamp format in CSV: %r", last)
    except Exception as e:
        log.warning("Could not read last timestamp from CSV: %s", e)
    return None


def write_record(record, columns):
    file_exists = os.path.isfile(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
            log.info("CSV created: %s  (%d columns)", CSV_FILE, len(columns))

        # Decode bytes keys so they match the string column names; columns not
        # in `columns` (RecNbr, discarded sensors) are dropped by extrasaction.
        writer.writerow(_decode_row(record))


# ---------------------------------------------------------------------------
# Main — outer reconnect loop + inner poll loop
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("  CR1000 Monitor  |  port=%s  baud=%d  table=%s  interval=%ds",
             SERIAL_PORT, BAUD_RATE, TABLE_NAME, POLL_INTERVAL)
    log.info("=" * 60)

    last_rec_nbr = None
    last_dt      = read_last_timestamp()
    if last_dt is not None:
        log.info("Resuming after last saved row at %s", last_dt)

    while True:
        device   = None
        tabledef = None
        columns  = None

        try:
            # ── Connect ────────────────────────────────────────────────────
            log.info("Connecting to %s @ %d baud...", SERIAL_PORT, BAUD_RATE)
            t0     = time.time()
            device = call_with_timeout(
                CR1000.from_url, CONNECT_TIMEOUT,
                f"serial:{SERIAL_PORT}:{BAUD_RATE}"
            )
            log.info("Connected in %.1fs", time.time() - t0)

            tabledef = call_with_timeout(fetch_tabledef, CALL_TIMEOUT, device)
            columns  = build_columns(tabledef)
            log.info("Column order (%d cols): %s ... %s",
                     len(columns), columns[:4], columns[-2:])

            # ── Poll loop ──────────────────────────────────────────────────
            while True:
                t0 = time.time()

                try:
                    recs = call_with_timeout(fetch_recent_records, CALL_TIMEOUT, device, tabledef)
                except Exception as e:
                    log.error("Fetch failed: %s — reconnecting", e)
                    log.debug(traceback.format_exc())
                    break   # exit inner loop → reconnect

                # Keep records newer than what we've already written (by RecNbr
                # within a run, by Timestamp across restarts), in ascending order.
                #
                # The batched fetch is NOT atomic: each batch requests the most
                # recent N records independently, so a batch delayed by a retry
                # can see a newer window than the others. Merging then leaves the
                # edge record covered by only some batches (a partial row). Only
                # emit records that every batch fully populated; stop at the first
                # incomplete one so we never write past a gap — it'll be complete
                # on the next poll.
                kept     = columns[1:]        # sensor columns (Timestamp excluded)
                expected = len(kept)
                new      = []
                deferred = None
                for r in recs:
                    rn, ts = r['_RecNbr'], r.get('Timestamp')
                    if ts is None:
                        continue
                    if last_rec_nbr is not None and rn <= last_rec_nbr:
                        continue
                    if last_dt is not None and ts <= last_dt:
                        continue
                    # Count only the kept columns; a dropped batch leaves some of
                    # them missing even though other (discarded) fields arrived.
                    row     = _decode_row(r)
                    nfields = sum(1 for c in kept if row.get(c) is not None)
                    if nfields < expected:
                        deferred = (rn, nfields)
                        break
                    new.append(r)

                if not recs:
                    log.warning("No record returned")
                elif not new:
                    if deferred:
                        log.info("Poll: record #%s incomplete (%d/%d fields) — deferring to next poll",
                                 deferred[0], deferred[1], expected)
                    else:
                        log.info("Poll: no new data (record #%s)", recs[-1]['_RecNbr'])
                else:
                    for r in new:
                        write_record(r, columns)
                        last_rec_nbr = r['_RecNbr']
                        last_dt      = r['Timestamp']
                    tail = ("  (deferring #%s: %d/%d fields)" % (deferred[0], deferred[1], expected)
                            if deferred else "")
                    log.info("Wrote %d new record(s)  |  latest #%s  |  %s  |  %.1fs fetch%s",
                             len(new), last_rec_nbr, last_dt, time.time() - t0, tail)

                # Sleep for the remainder of the poll interval
                elapsed = time.time() - t0
                sleep_s = max(0.0, POLL_INTERVAL - elapsed)
                log.debug("Sleeping %.1fs until next poll", sleep_s)
                time.sleep(sleep_s)

        except TimeoutError as e:
            log.error("TIMEOUT: %s", e)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            return
        except Exception as e:
            log.error("ERROR: %s", e)
            log.debug(traceback.format_exc())
        finally:
            if device:
                try:
                    device.bye()
                    log.debug("Bye sent")
                except Exception:
                    pass
                device = None

        log.info("Reconnecting in %ds...\n", RECONNECT_DELAY)
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    main()
