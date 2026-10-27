# Locomotion

Work for the **Mobility Trials** of the 2026 IEEE Humanoids Loco-Manipulation Challenge (Santa Clara, Dec 6–9, 2026), organized by NIST.

## Trial format

- Obstacles and rough terrain are in place. Five buttons along the course mark that the robot reached each key location. Each one must be pressed.
- The robot runs tasks A → D to the ending location, then repeats them in reverse back to the start. That is **one cycle**. Cycles repeat until time runs out.
- Trials last about **10–20 min**. Points come from the total number of tasks completed, and doing all tasks in order earns a bonus.
- Higher autonomy scores higher. Declare the autonomy level for each task. The apparatus can be pre-mapped on prep day.
- **Failures:** an operational failure (robot inoperable, needs a reset) costs points, and the clock keeps running during the reset. A failure that recovers without help (self-righting, auto-reboot) costs less. **Engaging the safety gantry counts as a balance failure.**

## Tasks

### A. Stepover obstacles

**Rules:** From the start, walk through the open doorway, turn right and press **button 1**, then turn around and press **button 2**. On the way, the robot must step over **two lengths of 4×4 lumber** (about 3.5 in × 3.5 in) lying across the walkway.

**Notes / strategy**

-

### B. Crossing ramps

**Rules:** An 8 ft × 8 ft section of **15° slopes** in a hill and pinwheel layout. Cross it to press **buttons 3 and 4**, which are at two of the manipulation task stations.

**Notes / strategy**

-

### C. Flat ground walking

**Rules:** Walk the **8 ft** flat section to the bottom of the stairs.

**Notes / strategy**

-

### D. Stairs ascend and descend

**Rules:** Climb a **half-flight of stairs** to the top landing, turn left, and press **button 5**. Then come back down.

**Notes / strategy**

-

### Bonus: Tote carrying

**Rules:** Do the mobility tasks while carrying a tote or milk crate. This is also part of the loco-manipulation phase (see [manipulation](../manipulation/README.md#loco-manipulation-phase-3-shared-with-locomotion)).

**Notes / strategy**

-

## Robot and operations constraints (all phases)

- One robot per team, **< 120 kg**, must fit through a **3 ft** doorway, and must run on battery.
- Required: secure carry handles, an attachment point for the overhead gantry rope, and an e-stop that someone is holding at all times.
- Operator: one at a time, at the operator station, using **only the system interface**. The operator may not watch the robot. Teammates may not talk to the operator unless there is a failure.
- Tethers are allowed. They should be spooled on the robot, or managed by a single cable handler who stays at the lane entrance. No sticks or other tools, and no routing over the lane walls.
