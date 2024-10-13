# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส
1.ชนันชิดา_6566

'''
#import library
from spatialmath import SE3
from HW3_utils import FKHW3
from math import pi
from typing import List

import numpy as np
import roboticstoolbox as rtb

d_1 = 0.0892
a_2 = 0.425
a_3 = 0.39243
d_4 = 0.109
d_5 = 0.093
d_6 = 0.082
q_initial = np.array([0.0, 0.0, 0.0])
q_singularity = np.array([0.0, 0.0, 3])
w_initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #(Fx, Fy, Fz, Tx, Ty, Tz)

robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha = 0.0     ,a = 0.0      ,d = d_1    ,offset = pi ),
        rtb.RevoluteMDH(alpha = pi/2    ,a = 0.0      ,d = 0.0    ,offset = 0.0),
        rtb.RevoluteMDH(alpha = 0.0     ,a = -a_2     ,d = 0.0    ,offset = 0.0),
    ],
    tool = SE3([
    [0, 0, -1, -(a_3 + d_6)],
    [0, 1, 0, -d_5],
    [1, 0, 0, d_4],
    [0, 0, 0, 1]]),
    name = "3DOF_Robot"
)
#=============================================<คำตอบข้อ 1>======================================================#
#code here
def endEffectorJacobianHW3(q:list[float])->list[float]:
    
    # เรียกใช้ฟังก์ชัน forward kinematics เพื่อรับค่าการหมุนแลพตำแหน่ง
    R, P, R_e, p_e = FKHW3(q)
    
    n = len(q)  # จำนวนข้อต่อ 
    
    # สร้างเมทริกซ์จาโคเบียนขนาด 6xn
    J = np.zeros((6, n))
    
    # กำหนดตำแหน่งของเฟรมที่ 0 (ฐาน)
    p0 = np.array([0, 0, 0])
    
    # วนลูปสำหรับแต่ละข้อต่อ
    for i in range(n):
        # ตำแหน่งของข้อต่อที่ i
        if i == 0:
            p_i = p0  # ถ้าเป็นข้อต่อแรก ใช้เฟรมฐาน
        else:
            p_i = P[:, i]  # ใช้ตำแหน่งข้อต่อที่ i-1
        
        # แกน z ของข้อต่อที่ i (แกนการหมุน)
        z_i = R[:, 2, i]  # แกน z ของเฟรมที่ i (คอลัมน์ที่สามของเมทริกซ์การหมุน)
        
        # คำนวณความเร็วเชิงเส้น (Translational Jacobian): Jv = z_i × (p_e - p_i)
        J[:3, i] = np.cross(z_i, (p_e - p_i))  # Cross product
        
        # คำนวณความเร็วเชิงมุม (Rotational Jacobian): Jw = z_i (สำหรับข้อต่อหมุน)
        J[3:, i] = z_i  # ความเร็วเชิงมุม
    
     
    # ใช้ค่า R_e เพื่อแปลง Jacobian ให้อยู่ในเฟรมของ end-effector (ถ้าจำเป็น)
    # แปลงส่วนของ rotational Jacobian เท่านั้น โดยใช้การ dot product กับ R_e
    J_e_translational = np.dot(R_e.T, J[:3, :])  # Transform the translational part
    J_e_rotational = np.dot(R_e.T, J[3:, :])    # Transform the rotational part

    J_e = np.vstack((J_e_translational, J_e_rotational))

    return J_e
# เรียกใช้ Jacobian ของหุ่นยนต์จากฟังก์ชันที่เราสร้างขึ้น
J_custom = endEffectorJacobianHW3(q_initial)

# เรียกใช้ Jacobian จาก roboticstoolbox
J_toolbox = robot.jacobe(q_initial)

# แสดงผลลัพธ์
#print("Jacobian ที่คำนวณได้จากฟังก์ชันของเรา:")
#print(J_custom)

#print("\nJacobian ที่ได้จาก Robotic Toolbox:")
#print(J_toolbox)
# คำนวณความแตกต่างระหว่าง Jacobian ที่ได้จากฟังก์ชันของเราและ Robotic Toolbox
J_difference = J_custom - J_toolbox

# แสดงผลลัพธ์ความแตกต่าง
#print("\nความแตกต่างระหว่าง Jacobian ที่คำนวณได้และ Robotic Toolbox:")
#print(J_difference)

#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:

    epsilon = 0.001  # ค่าเกณฑ์สำหรับการตัดสิน Singularity
    
    J = endEffectorJacobianHW3(q)  

    # ลดรูป Jacobian: ใช้เฉพาะส่วนของ Translational Jacobian (3x3)
    J_reduced = J[:3, :3]  

    # คำนวณค่า Determinant ของ Jacobian ที่ถูกลดรูป
    det_J = np.linalg.det(J_reduced)

    # แสดงค่า Determinant เพื่อการตรวจสอบ
    print(f"Determinant ของ Reduced Jacobian: {det_J}")

    # ตรวจสอบว่าค่า Determinant น้อยกว่าเกณฑ์ epsilon หรือไม่
    if abs(det_J) < epsilon:
        return 1  # อยู่ในสถานะ Singularity
    else:
        return 0  # ไม่อยู่ในสถานะ Singularity

# ทดสอบฟังก์ชัน
q_initial = [0.0, 0.0, 0.0]  # Configuration เริ่มต้น
flag = checkSingularityHW3(q_initial)
print(f"Singularity Flag (q_initial): {flag}")

q_singularity = [0.0, 0.0, 3]  # Configuration ที่อาจเป็น Singularity
flag = checkSingularityHW3(q_singularity)
print(f"Singularity Flag (q_singularity): {flag}")

#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:

    # เรียกใช้ฟังก์ชันที่คำนวณ Jacobian ของหุ่นยนต์
    J = endEffectorJacobianHW3(q)
    
    # คำนวณแรงบิดที่ข้อต่อ: tau = J^T * w
    tau = np.dot(J.T, w)
    
    return tau

# ทดสอบฟังก์ชัน
q_test = [0.0, np.pi/4, np.pi/4]  # ค่า configuration ของข้อต่อ
w_test = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # ค่า wrench (moment และ force)

tau_result = computeEffortHW3(q_test, w_test)
#print("Effort (Torque) at the joints:", tau_result)

#==============================================================================================================#

