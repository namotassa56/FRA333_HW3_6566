# file สำหรับตรวจคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส
1.ชนันชิดา_6566
'''
from FRA333_HW3_6566 import endEffectorJacobianHW3 , checkSingularityHW3 ,computeEffortHW3
from FRA333_HW3_6566 import checkSingularityHW3
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
q_singularity = np.array([0.0, pi/4, pi/4])

w_initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #(Fx, Fy, Fz, Tx, Ty, Tz)

robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha=0.0, a=0.0, d=d_1, offset=np.pi),
        rtb.RevoluteMDH(alpha=np.pi/2, a=0.0, d=0.0, offset=0.0),
        rtb.RevoluteMDH(alpha=0.0, a=-a_2, d=0.0, offset=0.0)
    ],
    tool=SE3([
        [0, 0, -1, -(a_3 + d_6)],
        [0, 1, 0, -d_5],
        [1, 0, 0, d_4],
        [0, 0, 0, 1]
    ]),
    name="3DOF_Robot"
)

#===========================================<ตรวจคำตอบข้อ 1>====================================================#
#code here
# เรียกใช้ Jacobian ของหุ่นยนต์จากฟังก์ชันที่เราสร้างขึ้น
J_custom = endEffectorJacobianHW3(q_initial)

# เรียกใช้ Jacobian จาก roboticstoolbox
J_toolbox = robot.jacobe(q_initial)

# แสดงผลลัพธ์
print("Jacobian ที่คำนวณได้จากฟังก์ชันของเรา:")
print(J_custom)

print("\nJacobian ที่ได้จาก Robotic Toolbox:")
print(J_toolbox)
# คำนวณความแตกต่างระหว่าง Jacobian ที่ได้จากฟังก์ชันของเราและ Robotic Toolbox
J_difference = J_custom - J_toolbox

# แสดงผลลัพธ์ความแตกต่าง
print("\nความแตกต่างระหว่าง Jacobian ที่คำนวณได้และ Robotic Toolbox:")
print(J_difference)
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 2>====================================================#
#code here
# ทดสอบฟังก์ชันด้วยค่า q_initial
q_initial = [0.0, 0.0, 0.0]
singularity_flag = checkSingularityHW3(q_initial)
print(f"Singularity Flag (q_initial): {singularity_flag}")

# ทดสอบฟังก์ชันด้วยค่า q_singularity
q_singularity = [0.0, np.pi/4, np.pi/4]
singularity_flag = checkSingularityHW3(q_singularity)
print(f"Singularity Flag (q_singularity): {singularity_flag}")
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 3>====================================================#
#code here

# ทดสอบฟังก์ชัน
q_test = [0.0, np.pi/4, np.pi/4]  # ค่า configuration ของข้อต่อ
w_test = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # ค่า wrench (moment และ force)

tau_result = computeEffortHW3(q_test, w_test)
print("Effort (Torque) at the joints:", tau_result)

#==============================================================================================================#