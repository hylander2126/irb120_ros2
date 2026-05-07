World, looking at it from side profile, +X to the right, +Z up, +Y into the page.

Regular Box-shaped object resting on surface. Object frame same orientation as world, but located at the TIPPING EDGE of the object.

TIPPING EDGE occurs at the object's smallest X-value and table contact. Essentially, the 'leftmost' part of the object.

Robot's frame defined exactly the same as world. 

F/T sensor mounted at EE, with the exact same orientation as world. Immediately after the sensor exists a rigid finger with a grippy rubber ball at the end. The ball is the only thing in contact with the object besides the table.

The goal is force tipping for objects with low friction. So the finger goes above the object frame and above the object and applies a downwards squash. The sensor reads a nearly perfectly +Z reading as 'reaction' to this squash.

Once sufficient force is reached, for now ~4N, the robot pulls towards itself, in the -X direction. The sensor now reads approximately +4N in Z and some quantity in +X as the 'reaction' to the pulling motion.

The robot maintains the squash force through a hybrid controller for some set time, then returns the object back to rest.

Our goal is to use the object pose, primarily its tipping angle, and measured Forces to estimate the mass and z_c com height that match the data to our simple torque balance model. Your job is to help me make this work. Confirmed working in mujoco simulation, however without hybrid squash controller, which is the key difference.
