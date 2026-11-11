"""Shared finite search geometry, not a physical-controller safety proof."""

# Pick approaches horizontally before opening and descending. The conservative
# gripper envelope extends 0.103 m below its centre, which is 0.055 m above the
# end-effector reference. It must clear the source during a lateral approach;
# the intended vertical grasp descent is exempt from that source envelope.
PICK_APPROACH_CLEARANCE = 0.048
PICK_APPROACH_XY_TOLERANCE = 0.002
