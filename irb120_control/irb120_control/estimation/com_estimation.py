import numpy as np
from irb120_control.estimation.helper_fns import axisangle2rot, rotvec_to_rot, Adjoint, TransInv, quat_to_rotvec

def rotvec_between(a, b):
    # rotation taking unit(a) -> unit(b), as a rotation vector (axis*angle)
    # a: (3,) reference vector; b: (N,3) query vectors
    a = np.atleast_2d(a) / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=-1)
    c = np.einsum('ni,ni->n', np.broadcast_to(a, b.shape), b)
    ang = np.arctan2(s, c)
    axis = axis / np.where(s[:, None] > 1e-9, s[:, None], 1.0)
    return axis * ang[:, None]  # (N,3) rotation vector

def construct_T(p, quat=None, rv=None):
    """Construct homogeneous transformation matrix from position and quaternion."""
    if quat is not None:
        R = rotvec_to_rot(quat_to_rotvec(quat))  # (N,3,3) rotation matrix from quaternion
    elif rv is not None:
        R = rotvec_to_rot(rv)  # (N,3,3) rotation matrix from rotation vector
    else:
        raise ValueError("Either 'quat' or 'rv' must be provided")
    T = np.zeros((len(p), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, 3] = 1.0
    return T


def model_bkwd_wrench(
    w_meas_S: np.ndarray,
    T_B_sensor: np.ndarray,
    T_B_obj: np.ndarray,
    # p_finger_O: np.ndarray,
) -> np.ndarray:
    """
    Compute the 'backward' applied wrench [tau, f] in object frame {O}.

    {O}, {B}, {S} are object, world/base, and sensor frames respectively.

    w_meas_S:   (N,6) measured wrenches in {S} — [tau, f] convention (Modern Robotics:
                moment first, force second) i.e. [tx ty tz fx fy fz]
    T_B_sensor: (N,4,4) sensor {S} poses in world frame {B}
    T_B_obj:    (N,4,4) object {O} poses in world frame {B}

    Returns: (N,6) applied wrench on object in {O}, [tau, f] convention [tx ty tz fx fy fz]
             Newton's 3rd law: reaction on object = negative of sensor reading.

    Uses: w_O = Ad_{T_SO}^T w_S  (Adjoint built for [tau, f] convention)
    """
    T_S_O = TransInv(T_B_sensor) @ T_B_obj          # (N,4,4) object pose in sensor frame
    AdT_S_O = Adjoint(T_S_O).reshape((-1, 6, 6)).transpose(0, 2, 1)   # (N,6,6)

    w_meas_O = np.einsum('nij,nj->ni', AdT_S_O, w_meas_S.reshape(-1, 6))   # (N,6) [tau,f] in {O}
    return -w_meas_O                                                          # (N,6) [tau,f] applied on object

def model_fwd_wrench(
        rot_vecs_B: np.ndarray,
        p_c_O: np.ndarray,
        mass: float,
):
    """
    Compute 'forward' gravity + ground reaction wrench in object frame {O}.

    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    rot_vecs_B: (N,3) array of axis-angle rotation vectors (angle in radians)
    p_c_O:      (N,3) position(s) of object CoM in object frame
    mass:       scalar mass of the object
    mu_table:   scalar friction coefficient of the table
    w_O_app:    (N,6) applied wrench in {O} in [tau, f] convention [tx ty tz fx fy fz]

    Returns: (w_O_grav, w_O_ground) where each is (N,6) in [tau, f] convention [tx ty tz fx fy fz]
    """
    rot_vecs_B = np.asarray(rot_vecs_B, dtype=float)
    R_B = rotvec_to_rot(rot_vecs_B)  # (N,3,3) object rotation in world frame
    R_B_T = R_B.transpose(0, 2, 1)  # (N,3b,3a) Transpose for inverse rotation (swaps correctly each 3x3 block)
    g_B = np.array([0, 0, -9.81])  # gravity in world/robot/table frame

    ## CONSTRUCT GRAVITY WRENCH IN OBJECT FRAME
    f_B_grav = mass * g_B                           # (3,) gravity force in world/robot/table frame
    f_O_grav = R_B_T @ f_B_grav                     # (N,3) gravity force in object frame
    tau_O_grav = -np.cross(p_c_O, f_O_grav)          # (N,3) gravity torque in object frame about CoM
    w_O_grav = np.hstack((tau_O_grav, f_O_grav))    # (N,6) gravity wrench [tau,f] in object frame
    
    return w_O_grav

def model_fwd_wrench_OLD(
        rot_vecs_B: np.ndarray,
        p_c_O: np.ndarray,
        mass: float,
        mu_table: float,
        w_O_app: np.ndarray = None
):
    """
    Compute 'forward' gravity + ground reaction wrench in object frame {O}.

    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    rot_vecs_B: (N,3) array of axis-angle rotation vectors (angle in radians)
    p_c_O:      (N,3) position(s) of object CoM in object frame
    mass:       scalar mass of the object
    mu_table:   scalar friction coefficient of the table
    w_O_app:    (N,6) applied wrench in {O} in [tau, f] convention [tx ty tz fx fy fz]

    Returns: (w_O_grav, w_O_ground) where each is (N,6) in [tau, f] convention [tx ty tz fx fy fz]
    """
    rot_vecs_B = np.asarray(rot_vecs_B, dtype=float)
    R_B = rotvec_to_rot(rot_vecs_B)  # (N,3,3) object rotation in world frame
    R_B_T = R_B.transpose(0, 2, 1)  # (N,3b,3a) Transpose for inverse rotation (swaps correctly each 3x3 block)
    g_B = np.array([0, 0, -9.81])  # gravity in world/robot/table frame

    if w_O_app is None:
        f_O_app = np.zeros((len(rot_vecs_B), 3), dtype=float)
    else:
        w_O_app = np.asarray(w_O_app, dtype=float)
        if w_O_app.ndim == 1 and w_O_app.shape[0] == 6:
            w_O_app = w_O_app.reshape(1, 6)
        f_O_app = w_O_app[:, 3:]                     # [tau,f] convention: force is in last 3 elements

    ## CONSTRUCT GRAVITY WRENCH IN OBJECT FRAME
    f_B_grav = mass * g_B                           # (3,) gravity force in world/robot/table frame
    f_O_grav = R_B_T @ f_B_grav                     # (N,3) gravity force in object frame
    tau_O_grav = np.cross(p_c_O, f_O_grav)          # (N,3) gravity torque in object frame about CoM
    w_O_grav = np.hstack((tau_O_grav, f_O_grav))    # (N,6) gravity wrench [tau,f] in object frame

    ## CONSTRUCT GROUND REACTION WRENCH IN OBJECT FRAME
    # 1. Get table normal force in object frame from force balance along table normal.
    n_B_table = np.array([0.0, 0.0, 1.0])
    n_O_table = np.einsum('nij,j->ni', R_B_T, n_B_table)  # (N,3)
    f_O_ext = f_O_grav + f_O_app # (N,3) total external force on object in object frame
    N_table_val = np.maximum(0.0, -np.einsum('ni,ni->n', f_O_ext, n_O_table)) # (N,) NOTE: negate ext force
    f_O_norm = np.einsum('n,ni->ni', N_table_val, n_O_table) # (N,3) table normal force vector in object frame

    # 2. Friction opposes the applied tangential force direction.
    # Use a capped magnitude per sample: min(mu*N, tangential force demand).
    # This captures static-like behavior below the Coulomb limit while preserving the Coulomb cap.
    f_O_app_tan = f_O_app - np.einsum('ni,ni->n', f_O_app, n_O_table)[:, None] * n_O_table
    tan_norm = np.linalg.norm(f_O_app_tan, axis=1)
    dir_fric_O = np.zeros_like(f_O_app_tan)
    valid = tan_norm > 1e-12
    dir_fric_O[valid] = -f_O_app_tan[valid] / tan_norm[valid, None]
    f_O_fric_max = mu_table * N_table_val
    f_O_fric_mag = np.minimum(f_O_fric_max, tan_norm)
    f_O_fr = np.einsum('n,ni->ni', f_O_fric_mag, dir_fric_O)
    
    # 3. Finish construction; ground cannot apply torque to object (explicit force)
    f_O_ground = f_O_norm + f_O_fr                              # (N,3) total ground reaction force in object frame
    t_O_ground = np.zeros_like(f_O_ground)                      # (N,3) ground reaction torque (zero: ground can't apply torque)
    w_O_ground = np.hstack((t_O_ground, f_O_ground))            # (N,6) ground reaction wrench [tau,f] in object frame

    # print("\nGravity wrench in object frame:\n", w_grav_O)
    # print("Ground reaction wrench in object frame:\n", w_O_ground)
    
    return w_O_grav, w_O_ground

# ============================================================================== #
# ========================= OLD MODELS  ========================= #
# ============================================================================== #

def tau_app_model(F, rf):
    """
    Compute torque about pivot due to applied force F at position rf.

    rf must be same shape as F (N, 3) and must account for object rotation.
    """
    # return np.cross(F, rf)
    tau = np.cross(rf, F)  # (N,3)
    return tau.ravel()


def tau_model(theta, m, zc, rc0_known, e_hat=[0,1,0]):
    """
    Compute the gravity torque given theta, mass, and z-height of CoM
    """
    W           = np.array([0, 0, -9.8067 * m]) # Weight in space frame
    # rc0_known   = np.array([-0.05, 0.0,  0.0]) # -0.05 , 0 , 0
    e_hat       = np.asarray(e_hat).flatten()  # ensure shape is (3,)
    rc0         = rc0_known.copy()
    rc0[2]      = zc
    theta       = np.asarray(theta).flatten()  # ensure shape is (n,)

    # TEMP testing new strategy
    # Get (batch) rotation matrix from axis-angle
    # -(rc0 x R(-theta)W)
    R = axisangle2rot(e_hat, -theta)   # (N,3,3)

    W_rotated = R @ W
    tau = -np.cross(rc0, W_rotated)  # (N,3)
    return tau.ravel()

## Force model (input is theta, output is force)
def F_model(theta, m, zc, rf, rc0_known, e_hat=[0,1,0]):
    """
    Force model: given angle(s) theta, mass m, CoM height zc, and
    per-sample lever arm rf (N,3) in the object frame, return the
    predicted contact force F(theta) in the object frame (N,3).

    theta : array-like, shape (N,) or (N,1)
    m     : mass
    zc    : CoM height above rc0_known.z
    rf    : lever arm from pivot to finger contact, shape (N,3)
    """
    theta = np.asarray(theta).reshape(-1)   # (N,)
    rf    = np.asarray(rf)                  # (N,3)
    N     = theta.shape[0]
    assert rf.shape == (N, 3), "rf must have shape (N,3)"

    g = 9.81
    # Geometry / axes in object frame
    e_hat     = np.asarray(e_hat).flatten()  # ensure shape is (3,)
    z_hat     = np.array([ 0.0, 0.0, 1.0])    # world/object z

    # CoM at height zc above rc0_known in z-direction
    rc0 = rc0_known.copy()
    rc0[2] = zc   # (3,)

    # 👉 Push direction in object frame (assumed constant)
    # Change to +1.0 if you push in +x in the object frame.
    d_hat = np.array([1.0, 0.0, 0.0])          # (3,)

    # Rotation matrices around e_hat by +theta and -theta
    R_pos = axisangle2rot(e_hat,  theta)        # (N,3,3)
    R_neg = axisangle2rot(e_hat, -theta)        # (N,3,3)

    # A(theta) = R_pos * (e × r_f)
    e_cross_rf = np.cross(e_hat, rf)            # (N,3)
    A = np.einsum('nij,nj->ni', R_pos, e_cross_rf)   # (N,3)

    # tmp(theta) = R_neg * (z × e)
    z_cross_ehat = np.cross(z_hat, e_hat)       # (3,)
    tmp = np.einsum('nij,j->ni', R_neg, z_cross_ehat)  # (N,3)

    # B(theta) = m g rc0ᵀ tmp  → (N,)
    B = m * g * (tmp @ rc0)

    # denom = Aᵀ d_hat = dot(A[i], d_hat), shape (N,)
    denom = A @ d_hat

    # alpha(theta) = B / (Aᵀ d_hat)
    alpha = B / denom                          # (N,)

    # F(theta) = alpha * d_hat  → (N,3)
    F_pred = alpha[:, None] * d_hat            # (N,3)

    return F_pred