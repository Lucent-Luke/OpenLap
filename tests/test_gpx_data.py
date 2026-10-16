from pathlib import Path

import pytest

from gpx_data import is_gpx, load_gpx
from exceptions import NoDataRowsError, MissingHeaderError

FIXTURES = Path(__file__).parent / "fixtures"
GPX_PATH          = str(FIXTURES / "sample.gpx")
GPX_NO_SPEED_PATH = str(FIXTURES / "sample_no_speed.gpx")
CSV_PATH          = str(FIXTURES / "racebox_car.csv")


# ── is_gpx ─────────────────────────────────────────────────────────────────────

def test_is_gpx_positive():
    assert is_gpx(GPX_PATH) is True


def test_is_gpx_negative_csv():
    assert is_gpx(CSV_PATH) is False


def test_is_gpx_negative_wrong_extension(tmp_path):
    f = tmp_path / "track.txt"
    f.write_text("<gpx>...</gpx>")
    assert is_gpx(str(f)) is False


def test_is_gpx_nonexistent():
    assert is_gpx('/nonexistent/file.gpx') is False


# ── load_gpx — source and structure ───────────────────────────────────────────

def test_load_gpx_source():
    session = load_gpx(GPX_PATH)
    assert session.source == 'GPX'


def test_load_gpx_has_points():
    session = load_gpx(GPX_PATH)
    assert len(session.all_points) == 8


def test_load_gpx_has_one_lap():
    session = load_gpx(GPX_PATH)
    assert len(session.laps) == 1
    assert session.laps[0].lap_num == 1


def test_load_gpx_source_speed_unit():
    session = load_gpx(GPX_PATH)
    assert session.source_speed_unit == 'kmh'


def test_load_gpx_lap_is_not_outlap():
    session = load_gpx(GPX_PATH)
    assert session.laps[0].is_outlap is False


def test_load_gpx_track_name():
    session = load_gpx(GPX_PATH)
    assert 'Spa' in session.track or session.track != ''


def test_load_gpx_date_utc():
    session = load_gpx(GPX_PATH)
    assert '2024-06-15' in session.date_utc


def test_load_gpx_not_bike():
    session = load_gpx(GPX_PATH)
    assert session.is_bike is False


# ── Speed ──────────────────────────────────────────────────────────────────────

def test_load_gpx_speed_from_extension():
    # Fixture has extension speed — peak should reflect the 30 m/s → 108 km/h point
    session = load_gpx(GPX_PATH)
    max_speed = max(p.speed for p in session.all_points)
    assert max_speed > 50.0   # 30 m/s * 3.6 = 108 km/h, smoothed down somewhat


def test_load_gpx_speed_non_negative():
    session = load_gpx(GPX_PATH)
    for pt in session.all_points:
        assert pt.speed >= 0.0


def test_load_gpx_derived_speed():
    # File with no extension speed — speed must still be derived and non-negative
    session = load_gpx(GPX_NO_SPEED_PATH)
    speeds = [p.speed for p in session.all_points]
    assert all(s >= 0.0 for s in speeds)
    assert max(speeds) > 0.0   # moving track — should have positive speed


# ── Elapsed time ───────────────────────────────────────────────────────────────

def test_load_gpx_elapsed_starts_at_zero():
    session = load_gpx(GPX_PATH)
    assert session.all_points[0].elapsed == pytest.approx(0.0)


def test_load_gpx_elapsed_monotonic():
    session = load_gpx(GPX_PATH)
    pts = session.all_points
    for a, b in zip(pts, pts[1:]):
        assert b.elapsed >= a.elapsed


def test_load_gpx_duration():
    session = load_gpx(GPX_PATH)
    # Fixture spans 7 seconds (13:00:00 to 13:00:07)
    assert session.laps[0].duration == pytest.approx(7.0, abs=0.1)


# ── Derived G-forces ───────────────────────────────────────────────────────────

def test_load_gpx_gforce_x_range():
    session = load_gpx(GPX_PATH)
    for pt in session.all_points:
        assert -5.0 <= pt.gforce_x <= 5.0


def test_load_gpx_gforce_y_range():
    session = load_gpx(GPX_PATH)
    for pt in session.all_points:
        assert -5.0 <= pt.gforce_y <= 5.0


# ── interpolate_at works with GPX session ─────────────────────────────────────

def test_load_gpx_interpolate_at_midpoint():
    session = load_gpx(GPX_PATH)
    pts = session.all_points
    mid_t = (pts[2].elapsed + pts[3].elapsed) / 2
    result = session.interpolate_at(mid_t)
    assert result is not None


# ── No-timestamp fallback (1 Hz synthesis) ────────────────────────────────────

def _write_gpx_no_time(tmp_path, n_points=5):
    pts = ''.join(
        f'<trkpt lat="50.{i:04d}" lon="4.{i:04d}"></trkpt>' for i in range(n_points)
    )
    content = (
        '<?xml version="1.0"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f'<trk><trkseg>{pts}</trkseg></trk></gpx>'
    )
    p = tmp_path / "no_time.gpx"
    p.write_text(content)
    return str(p)


def test_load_gpx_no_timestamps_assigns_distinct_elapsed(tmp_path):
    path = _write_gpx_no_time(tmp_path, n_points=5)
    session = load_gpx(path)
    elapsed = [p.elapsed for p in session.all_points]
    # Every point must have a distinct, strictly increasing elapsed time
    assert elapsed == sorted(set(elapsed))
    assert len(set(elapsed)) == len(elapsed)


def test_load_gpx_no_timestamps_1hz_spacing(tmp_path):
    path = _write_gpx_no_time(tmp_path, n_points=5)
    session = load_gpx(path)
    elapsed = [p.elapsed for p in session.all_points]
    assert elapsed == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


# ── Sensor channels (heart rate, cadence, power) ──────────────────────────────

def _write_gpx_sensors(tmp_path, point_exts, ns=''):
    """One track point per entry in *point_exts* (the inner XML of its
    <extensions>, or None for no extensions), 1 s apart."""
    pts = ''
    for i, ext in enumerate(point_exts):
        ext_xml = f'<extensions>{ext}</extensions>' if ext is not None else ''
        pts += (f'<trkpt lat="50.{i:04d}" lon="4.{i:04d}">'
                f'<time>2024-06-15T13:00:{i:02d}Z</time>{ext_xml}</trkpt>')
    content = (
        '<?xml version="1.0"?>'
        f'<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1" {ns}>'
        f'<trk><trkseg>{pts}</trkseg></trk></gpx>'
    )
    p = tmp_path / "ride.gpx"
    p.write_text(content)
    return str(p)


_GARMIN_NS = ('xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" '
              'xmlns:pwr="http://www.garmin.com/xmlschemas/PowerExtension/v1"')


def test_load_gpx_garmin_sensor_channels(tmp_path):
    exts = [
        f'<pwr:PowerInWatts>{200 + i}</pwr:PowerInWatts>'
        f'<gpxtpx:TrackPointExtension><gpxtpx:hr>{140 + i}</gpxtpx:hr>'
        f'<gpxtpx:cad>{85 + i}</gpxtpx:cad></gpxtpx:TrackPointExtension>'
        for i in range(4)
    ]
    session = load_gpx(_write_gpx_sensors(tmp_path, exts, _GARMIN_NS))
    assert session.extra_channel_meta == {
        'Heart Rate': {'label': 'Heart Rate', 'unit': 'bpm'},
        'Cadence':    {'label': 'Cadence',    'unit': 'rpm'},
        'Power':      {'label': 'Power',      'unit': 'W'},
    }
    assert [p.extra['Heart Rate'] for p in session.all_points] == [140, 141, 142, 143]
    assert [p.extra['Cadence'] for p in session.all_points] == [85, 86, 87, 88]
    assert [p.extra['Power'] for p in session.all_points] == [200, 201, 202, 203]


def test_load_gpx_strava_power(tmp_path):
    # Strava writes power as a bare, un-namespaced <power> in <extensions>
    exts = [f'<power>{250 + 10 * i}</power>' for i in range(3)]
    session = load_gpx(_write_gpx_sensors(tmp_path, exts))
    assert list(session.extra_channel_meta) == ['Power']
    assert [p.extra['Power'] for p in session.all_points] == [250, 260, 270]


def test_load_gpx_sensor_gap_interpolated(tmp_path):
    exts = ['<hr>100</hr>', None, '<hr>120</hr>', '<hr>not a number</hr>', '<hr>140</hr>']
    session = load_gpx(_write_gpx_sensors(tmp_path, exts))
    assert [p.extra['Heart Rate'] for p in session.all_points] == pytest.approx(
        [100, 110, 120, 130, 140])


def test_load_gpx_without_sensors_has_no_extra_channels():
    session = load_gpx(GPX_PATH)
    assert session.extra_channel_meta == {}
    assert all(p.extra == {} for p in session.all_points)


def test_load_gpx_sensor_channels_discoverable_and_interpolated(tmp_path):
    from channel_discovery import list_channels
    exts = [f'<hr>{100 + 10 * i}</hr><power>{200 + 20 * i}</power>' for i in range(10)]
    session = load_gpx(_write_gpx_sensors(tmp_path, exts))
    keys = {c['key'] for c in list_channels(session)}
    assert {'Heart Rate', 'Power'} <= keys
    mid = session.interpolate_at(0.5)
    assert mid.extra['Heart Rate'] == pytest.approx(105)
    assert mid.extra['Power'] == pytest.approx(210)


# ── Error cases ────────────────────────────────────────────────────────────────

def test_load_gpx_empty_trkseg_raises(tmp_path):
    bad = tmp_path / "empty.gpx"
    bad.write_text('<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg></trkseg></trk></gpx>')
    with pytest.raises(NoDataRowsError):
        load_gpx(str(bad))


def test_load_gpx_corrupt_xml_raises(tmp_path):
    bad = tmp_path / "corrupt.gpx"
    bad.write_text('<gpx><this is not valid xml')
    with pytest.raises(MissingHeaderError):
        load_gpx(str(bad))


# ── session_scanner integration ────────────────────────────────────────────────

def test_scan_csvs_finds_gpx(tmp_path):
    from session_scanner import scan_csvs
    import shutil
    shutil.copy(GPX_PATH, tmp_path / "track.gpx")
    results = scan_csvs(str(tmp_path))
    gpx_results = [r for r in results if r.endswith('.gpx')]
    assert len(gpx_results) == 1


def test_csv_source_gpx():
    from session_scanner import _csv_source
    assert _csv_source(GPX_PATH) == 'GPX'


def test_read_csv_start_time_gpx():
    from session_scanner import _read_csv_start_time
    dt = _read_csv_start_time(GPX_PATH)
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 6
    assert dt.tzinfo is not None
