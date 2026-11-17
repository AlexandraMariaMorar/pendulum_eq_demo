physical_data: cleaned Tracker recordings, one file per release
================================================================

Files
  mass090g_exp1..4.csv   90 g bob   (from data_dirty_1.txt)
  mass136g_exp1..2.csv   136 g bob  (from data_dirty_2.1.txt; 2.2 was an identical copy)
  overview.png           angle against time for every file
  make_physical_data.py  the script that made these from the original Tracker exports
                         (the raw exports are not in the repository; Stephen keeps them locally)

Columns (comma separated, lines starting with # are notes)
  t_s          time since the start of this release, in seconds (corrected, see below)
  x_m, y_m     bob position in metres, origin at the pivot as set in Tracker
  theta_rad    angle from straight down, positive to the right (pivot fitted per recording)
  frame        frame number in the original video
  t_tracker_s  Tracker's original time, kept for reference (uses the wrong frame rate)
t, x, y are the first three columns, so the existing loaders read these files directly.

What was changed
  1. Time axis. Tracker assumed 60 fps (90 g) and 30 fps (136 g). With those times the
     swings imply g of about 60 and 14 m/s^2. Rebuilt from the frame number instead:
     90 g at 24 fps, 136 g at 25 fps (nearest standard rates; the data alone point to
     24.2 and 24.8 fps). This is an INFERENCE. Check the video's real frame rate
     (Windows: right-click > Properties > Details > Frame rate). If it differs, change
     RECORDINGS at the top of make_physical_data.py and rerun it on the raw exports.
  2. Filled-in frames removed. Where Tracker lost the bob it drew straight lines between
     marks; 1705 of the 2359 rows in the 90 g file were like that. Detected as frames
     whose x and y change perfectly smoothly (second difference < 0.1 mm).
  3. Split by release. Each piece starts after a fill or a time gap, and the swing angle
     jumps up between pieces, so each is a separate release. Time restarts at 0 in each.
  Nothing else was changed: no smoothing, no outlier removal.

Check after correction (swing period, corrected for large angles, against measured length)
file                 frames length_s swings period_s angle_deg    L_m g_check
mass090g_exp1.csv        52     2.12      1    1.217        56  0.319    9.61
mass090g_exp2.csv       349    14.50     11    1.179        38  0.322    9.67
mass090g_exp3.csv       112     4.62      3    1.204        59  0.312    9.73
mass090g_exp4.csv       141     5.83      4    1.221        59  0.321    9.73
mass136g_exp1.csv       369    14.72     10    1.337        49  0.410    9.94
mass136g_exp2.csv       204     8.12      5    1.390        64  0.414    9.94
  g_check should be about 9.81. Values of 9.6-9.9 mean the corrected time axis is right to
  within about 1-2 %. With the original times the same column read about 60 and 14.

Known limitations
  - mass090g_exp1 is a single swing (2 s): too short to fit an equation, kept for completeness.
  - Some frames sit 3-10 % off the circle (tracking slips near the turning points, and at the
    end of mass136g_exp1, around 9.5-11.5 s). Left in; the smoothing step of the analysis
    handles small slips, but trim that end if a fit looks odd.
  - Releases last 2-15 s, so the slow fading of the swing (damping) is poorly measured.
    Comparing the two masses' damping needs longer unbroken recordings (30 s or more).
