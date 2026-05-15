import numpy as np
from irb120_control.estimation.helper_fns import axisangle2rot, rotvec_to_rot, Adjoint, TransInv, quat_to_rotvec


def compute_applied_wrench(
    f_meas_S: np.ndarray,
    Q_ft: np.ndarray,
    pitch_contact: np.ndarray,
    p_finger_O: np.ndarray,
) -> np.ndarray:
    """Compute applied wrench in object frame directly from logged quantities.

    Skips 4x4 homogeneous matrix packing. Equivalent to model_bkwd_wrench.

    Steps:
      R_obj = Ry(pitch)                     object frame orientation in base
      R_O_S = R_obj.T @ R_ft               sensor rotation expressed in object frame
      f_app_O = -(R_O_S @ f_S)             Newton's 3rd law
      tau_app_O = p_finger_O × f_app_O    torque about object origin (pivot)

    Args:
        f_meas_S:      (N,3) measured forces in sensor frame
        Q_ft:          (N,4) ft_link quaternion [x,y,z,w] in base frame
        pitch_contact: (N,) tipping angle in radians (rotation about Y)
        p_finger_O:    (N,3) finger position in object frame (constant, no-slip)
    """
    N = len(pitch_contact)
    
    R_obj = axisangle2rot(np.array([0, 1, 0]), pitch_contact)   # (N,3,3) R_B_O object rotation in {B} (robot/world)
    R_ft  = rotvec_to_rot(quat_to_rotvec(Q_ft))                 # (N,3,3) R_B_S      ft rotation in {B}

    # R_O_S = R_obj.T @ R_ft
    R_O_S = np.einsum('nji,njk->nik', R_obj, R_ft)            # (N,3,3) {S} in {O}

    f_app_O = -np.einsum('nij,nj->ni', R_O_S, f_meas_S)      # (N,3)
    t_app_O = np.cross(p_finger_O, f_app_O)                  # (N,3)

    return np.hstack((f_app_O, t_app_O))


def compute_applied_wrench_surface(
    f_meas_S: np.ndarray,
    Q_ft: np.ndarray,
    pitch_contact: np.ndarray,
    p_finger_O: np.ndarray,
) -> tuple:
    """Compute applied wrench in object frame using contact surface geometry.

    The EE maintains constant world orientation throughout squashing, so R_B_S
    is time-varying only due to the fixed sensor mount offset — captured by Q_ft.
    f_meas_S is first rotated to the world frame via Q_ft, then decomposed into
    normal/tangential components at the (tilting) contact surface, then rotated
    into the object frame.

    Surface assumption: top-face normal is [0,0,1] in world at θ=0. As the
    object tips by θ (Ry(θ)), the normal in world frame becomes:
        n_B(θ) = [sin(θ), 0, cos(θ)]

    Args:
        f_meas_S:      (N,3) measured forces in sensor frame
        Q_ft:          (N,4) sensor quaternion [x,y,z,w] in world/base frame
        pitch_contact: (N,) tipping angle in radians (rotation about +Y)
        p_finger_O:    (3,) contact point in object frame (constant, no-slip)

    Returns:
        w_app_O:   (N,6) applied wrench [fx,fy,fz,tx,ty,tz] in object frame
        f_norm_O:  (N,3) normal component of applied force in object frame
        f_tan_O:   (N,3) tangential component of applied force in object frame
    """
    N = len(pitch_contact)

    # Rotate measured force from sensor frame into world frame using Q_ft
    R_B_S = rotvec_to_rot(quat_to_rotvec(Q_ft))                # (N,3,3) R_B_S
    f_meas_W = np.einsum('nij,nj->ni', R_B_S, f_meas_S)        # (N,3) force in world frame

    # Surface normal in world frame as object tips: Ry(θ) @ [0,0,1]
    n_B = np.column_stack([np.sin(pitch_contact),
                           np.zeros(N),
                           np.cos(pitch_contact)])              # (N,3)

    # Decompose world-frame force into normal and tangential at the contact surface
    f_dot_n  = np.einsum('ni,ni->n', f_meas_W, n_B)            # (N,) scalar projection onto normal
    f_norm_B = np.einsum('n,ni->ni', f_dot_n, n_B)             # (N,3) normal component in world
    f_tan_B  = f_meas_W - f_norm_B                             # (N,3) tangential component in world

    # Rotate both components from world frame into object frame (R_O_B = R_B_O.T)
    rv_contact = np.zeros((N, 3)); rv_contact[:, 1] = pitch_contact
    R_obj = rotvec_to_rot(rv_contact)                           # (N,3,3) R_B_O
    f_norm_O = np.einsum('nji,nj->ni', R_obj, f_norm_B)        # (N,3) normal in {O}
    f_tan_O  = np.einsum('nji,nj->ni', R_obj, f_tan_B)         # (N,3) tangential in {O}

    # Applied force on object = Newton's 3rd law
    f_app_O = -(f_norm_O + f_tan_O)                            # (N,3)
    t_app_O = np.cross(p_finger_O, f_app_O)                    # (N,3)

    w_app_O = np.hstack((f_app_O, t_app_O))                    # (N,6)
    return w_app_O, -f_norm_O, -f_tan_O                        # sign: forces ON the object


def model_bkwd_wrench(
    w_meas_S: np.ndarray,
    T_B_sensor: np.ndarray,
    T_B_obj: np.ndarray,
    p_finger_O: np.ndarray,
) -> np.ndarray:
    """
    Compute the 'backward' applied wrench [f; tau] in object frame {O}.

    {O}, {B}, {S} are object, world/base, and sensor frames respectively.

    w_meas_S:   (N,6) measured wrenches in {S}  [fx fy fz tx ty tz]
    T_B_sensor: (N,4,4) sensor poses in world frame
    T_B_obj:    (N,4,4) object poses in world frame
    p_finger_O: (N,3) or (3,) contact-point position in {O}
    """
    # First get sensor pose in object frame, then get corresponding AdT
    T_O_S = TransInv(T_B_obj) @ T_B_sensor   # (N,4,4) sensor pose in object frame
    AdT_S_O = Adjoint(T_O_S).reshape((-1, 6, 6))          # (N,6,6)
    w_meas_S = w_meas_S.reshape(-1, 6)                    # (N,6) measured wrench in sensor frame
    # print(AdT_S_O.shape, w_meas_S.shape)
    w_meas_O = np.einsum('nij,nj->ni', AdT_S_O, w_meas_S.reshape(-1,6))    # (N,6) wrench in {O}

    f_app_O  = -w_meas_O[:, :3]                               # Newton's 3rd law
    t_app_O  = np.cross(p_finger_O, f_app_O)                  # r × f about object origin
    w_app_O = np.hstack((f_app_O, t_app_O))                      # (N,6)
    return w_app_O


def model_fwd_wrench(
        rot_vecs_B: np.ndarray,
        p_c_O: np.ndarray,
        mass: float,
        mu_table: float,
        w_O_app: np.ndarray = None
):
    """
    Compute 'forward' gravity + ground reaction wrench [F; tau] IN OBJECT FRAME
    {O}, {B}, {S} are object, robot base/table/world, and sensor frames, respectively.

    rot_vecs: (N,3) array of axis-angle rotation vectors (angle in radians)

    w_O_app: (N,6) array of applied wrenches in object frame (F_x, F_y, F_z, tau_x, tau_y, tau_z)

    p_c_O: (N,3) position(s) of object CoM in object frame. N samples for liquid-filled containers.

    mass: scalar mass of the object

    mu_table: scalar friction coefficient of the table

    Returns: (w_O_grav, w_O_ground) where each is a (N,6) array of wrenches in object frame.
    """
    rot_vecs_B = np.asarray(rot_vecs_B, dtype=float)
    R_B = rotvec_to_rot(rot_vecs_B)  # (N,3,3) object rotation in world frame
    R_B_T = R_B.transpose(0, 2, 1)  # (N,3b,3a) Transpose for inverse rotation (swaps correctly each 3x3 block)
    g_B = np.array([0, 0, -9.81])  # gravity in world/robot/table frame
    n_samples = rot_vecs_B.shape[0]

    if w_O_app is None:
        f_O_app = np.zeros((n_samples, 3), dtype=float)
    else:
        w_O_app = np.asarray(w_O_app, dtype=float)
        if w_O_app.ndim == 1 and w_O_app.shape[0] == 6:
            w_O_app = w_O_app.reshape(1, 6)
        f_O_app = w_O_app[:, :3]
        
    ## CONSTRUCT GRAVITY WRENCH IN OBJECT FRAME
    f_B_grav = mass * g_B                           # (3,) gravity force in world/robot/table frame
    f_O_grav = R_B_T @ f_B_grav                     # (N,3) gravity force in object frame
    tau_O_grav = np.cross(p_c_O, f_O_grav)          # (N,3) gravity torque in object frame about CoM
    w_O_grav = np.hstack((f_O_grav, tau_O_grav))    # (N,6) gravity wrench in object frame

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
    t_O_ground = np.zeros_like(f_O_ground)                      # (N,3) ground reaction torque in object frame (assumed zero since ground cannot apply torque)
    w_O_ground = np.hstack((f_O_ground, t_O_ground))            # (N,6) ground reaction wrench in object frame

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