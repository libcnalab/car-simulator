# Known issues found by cna-engine

Defects found by cna-engine's agent while running this game's suite against the
engine's pinned dependencies and comparing its sun with the engine's
(2026-10-20/21). They are not fixed here; each entry has a suggested fix. Their
source record is `upstream-issues.md` in
[libcna/cna-engine](https://github.com/libcna/cna-engine) (UP numbers below).
Checked against `main` at `77c36ee` on 2026-10-22.

## The map-load budget fails in unoptimized GCC/Clang builds (UP-009)

`tests/Map/SampleMapTests.cpp` (`SampleMap.LipovaLoadsAndIsFullyConnected`,
around line 82) asserts `loadSeconds < 6.0`, the production loading budget. It
relaxes the bound for sanitizer builds (30 s) and MSVC Debug (60 s), but not for
unoptimized GCC/Clang builds: in a Debug build on a loaded machine (load average
about 20) the map took 16.0 s, so the test failed although every other assertion
held and the other 320 tests passed. The same fixture loads in about 15.6 s in
each of the following tests.

Suggested fix: relax the bound for any build without `NDEBUG`, or move the
budget into a Release-only benchmark, so the suite's result does not depend on
the build type and the machine's load.

## The sun's azimuth is mirrored between 00:00 and 01:00 (UP-016)

In `simulator/src/Render/LightingRig.cpp` (`LightingRig::SetTimeOfDay`) the hour
angle is `(hours - 13) * 15` degrees and is not wrapped, so before 01:00 it is
below -180 degrees; the azimuth is `acos(...)` mirrored to the west only when
the hour angle is positive, so for those hours it lands on the wrong side of
north. At 49.8 degrees north and a declination of 20 degrees the azimuth runs
338 degrees at 23:30 and 345 at 23:59, then 15 at 00:01 and back to 0 at 01:00,
instead of on through 352 to 360. The sun is below the horizon then, but the
night key light (the moon) is derived from the same direction, so the scene's
light turns by about 30 degrees at midnight and back over the following hour.

Suggested fix: wrap the hour angle into [-180, 180) before the azimuth, or use
the two-argument form `atan2(-cos d sin H, cos l sin d - sin l cos d cos H)`,
which needs no branch (cna-engine's `CnaEngine::SunAnglesFor` does so).
