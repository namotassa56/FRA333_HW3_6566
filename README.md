# FRA333_HW3_6566

## ผู้จัดทำ
- **ชื่อ:ชนันชิดา  โปร่งจิต
- **รหัสนักศึกษา:65340500066

---

## ภาพรวมของโปรเจกต์
โปรเจกต์นี้เกี่ยวข้องกับ **Kinematics and Robotics** โดยเน้นการคำนวณ Jacobian, การตรวจสอบ Singularity และการคำนวณแรงบิด (Effort) ที่ข้อต่อหุ่นยนต์ 3-DOF (RRR).
## RRR Robot (3-DOF Robot)
หุ่นยนต์ RRR ที่มีข้อต่อแบบหมุน (Revolute Joint) จำนวน 3 ข้อ โดยแต่ละข้อมีมุมหมุนอิสระ. การเคลื่อนที่ของหุ่นยนต์ใน 3 มิตินี้เรียกว่า **Degrees of Freedom (DOF)**.
หุ่นยนต์ RRR นี้เหมาะสำหรับงานที่ต้องการความยืดหยุ่นในการเข้าถึงวัตถุในหลายทิศทาง เช่น หุ่นยนต์หยิบจับวัตถุ.
### DH-Table และโครงสร้างของหุ่นยนต์ RRR
![DH-Table และโครงสร้าง](image.png)
---

## Setup Environment
   ```bash
-  pip install roboticstoolbox-python spatialmath
   ```
-  Python 3.9.x or newer**
-  `numpy`: For numerical operations.
-  `Robotics Toolbox for Python`**: To work with robot models and calculate kinematics.
-  `math`**: For mathematical calculations.
-  `spatialmath`**: For spatial transformations.
-  `matplotlib`**: For plotting graphs.

---

## Part DH-Table
    จะเลือกใช้การดู Frame ด้วยการทำ DH-Table จะได้การเคลื่อนที่ในเเต่ละเฟรม ได้ดังนี้
		
		จาก Frame 0 -> 1 :
		
			มีการเคลื่อนที่ในเเกน Z ด้วยระยะ d_1 เเละ มีการหมุน 180 องศา
	
		จาก Frame 1 -> 2 :
		
			มีการเคลื่อนที่ในเเกน X ด้วยระยะ 0 เเละ มีการหมุน 90 องศา
			
		จาก Frame 2 -> e :
		
			มีการเคลื่อนที่ในเเกน X ด้วยระยะ a_2 ในทิศสวนทาง 
    สามารถสร้างตาราง  DH-Table ได้ดังนี้
    
		┌────────┬───────┬───────┬────────┐
		│  aⱼ₋₁  │ ⍺ⱼ-1  │   θⱼ  │    dⱼ  │
		├────────┼───────┼───────┼────────┤
		│    0.0 │  0.0° │  180° │   d_1  │
		│    0.0 │  90.0°│  0.0  │    0.0 │
		│   -a_2 │  0.0° │  0.0  │    0.0 │
		└────────┴───────┴───────┴────────┘


## Part Transformation

	การเคลื่อนย้ายจากตำแหน่ง Frame ที่ 3 ไปยังจุดสุดท้าย End-Effector

        การเคลื่อนที่เกิดขึ้นภายในเเกน X :

                Trans_X = a_3 + d_6 

        การเคลื่อนที่เกิดขึ้นภายในเเกน Y :
                
                Trans_Y = -d_5

        การเคลื่อนที่เกิดขึ้นภายในเเกน Z :

                Trans_Z = d_4

        มีการหมุนของเเกน Y ในสู่ตำแหน่งEnd-Effector เท่ากับ

                Rot_Y = -90 degree

        นำมาทำ Transformation Matrix สู่ตำเเหน่ง End-Effector

## การสร้างโมเดลหุ่นยนต์ 3-DOF (RRR Robot)
    -**สร้างโมเดลหุ่นยนต์ RRR ด้วยการใช้ Denavit-Hartenberg (DH) Parameters.
    -**ใช้ฟังก์ชัน RevoluteMDH จาก roboticstoolbox เพื่อกำหนดข้อต่อแบบหมุน (revolute joints).
    -**พารามิเตอร์ tool กำหนดตำแหน่งของปลายมือ (end-effector).
    ```bash
-  robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(alpha=0.0, a=0.0, d=d_1, offset=pi),
        rtb.RevoluteMDH(alpha=pi/2, a=0.0, d=0.0, offset=0.0),
        rtb.RevoluteMDH(alpha=0.0, a=-a_2, d=0.0, offset=0.0),
    ],
    tool=SE3([
        [0, 0, -1, -(a_3 + d_6)],
        [0, 1, 0, -d_5],
        [1, 0, 0, d_4],
        [0, 0, 0, 1]
    ]),
    name="3DOF_Robot"
)
    ```

# Function
## 1. การคำนวณ Jacobian: endEffectorJacobianHW3(q)
   Input:
    -q: List ของมุมข้อต่อในหน่วยเรเดียน เช่น [q1, q2, q3]
   Output:
    -J_e: Jacobian Matrix 

### สมการที่เกี่ยวข้อง

### Jacobian Matrix $$ \mathbf{J} $$
Jacobian Matrix แสดงความสัมพันธ์ระหว่าง **ความเร็วเชิงมุมของข้อต่อ** กับ **ความเร็วเชิงเส้นและความเร็วเชิงมุมของปลายมือหุ่นยนต์ (End-effector)**.

$$
\mathbf{J} = 
\begin{bmatrix}
\mathbf{J}_v \\ 
\mathbf{J}_\omega
\end{bmatrix}
$$


- $$ \mathbf{J}_v $$: ใช้แปลงความเร็วเชิงมุมของข้อต่อให้เป็น **ความเร็วเชิงเส้น** ของปลายมือหุ่นยนต์
- $$ \mathbf{J}_\omega $$: ใช้แปลงความเร็วเชิงมุมของข้อต่อให้เป็น **ความเร็วเชิงมุม** ของปลายมือหุ่นยนต์

---

### ความเร็วเชิงเส้น

$$
\mathbf{v}_i = \mathbf{z}_i \times (\mathbf{p}_e - \mathbf{p}_i)
\]

- \( \mathbf{z}_i \): แกนหมุนของข้อต่อ \(i\) (แกน \(z\) ของเฟรมท้องถิ่น)  
- \( \mathbf{p}_e \): ตำแหน่งของปลายมือหุ่นยนต์  
- \( \mathbf{p}_i \): ตำแหน่งของข้อต่อ \(i\)  
- \( \times \): การคูณแบบ **Cross Product**

$$
---

### ความเร็วเชิงมุม

$$
\mathbf{\omega}_i = \mathbf{z}_i
\]

- ความเร็วเชิงมุมของข้อต่อ \(i\) เท่ากับทิศทางของแกน \(z\) ของข้อต่อนั้น
$$
---
การคำนวณ Jacobian ใน Python

```bash
    def endEffectorJacobianHW3(q:list[float])->list[float]:
    
    # เรียกใช้ฟังก์ชัน forward kinematics เพื่อรับค่าการหมุนและตำแหน่ง
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
    
     
    # แปลงส่วนของ rotational Jacobian เท่านั้น โดยใช้การ dot product กับ R_e
    J_e_translational = np.dot(R_e.T, J[:3, :])  # Transform the translational part
    J_e_rotational = np.dot(R_e.T, J[3:, :])    # Transform the rotational part

    J_e = np.vstack((J_e_translational, J_e_rotational))

    return J_e
    ```
    คำอธิบาย :
    -**ใช้ฟังก์ชัน FKHW3 เพื่อรับตำแหน่งและการหมุนของหุ่นยนต์.
    -**Cross Product ใช้สำหรับคำนวณความเร็วเชิงเส้นใน Jacobian.
    -**Rotation Matrix R_e ใช้สำหรับแปลงเฟรม Jacobian ให้อยู่ในเฟรมของปลายมือ (end-effector).

    ```
    Test :

```bash
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
    
    ```

    ได้ผลลัพธ์ดังนี้ :
#### Jacobian ที่คำนวณได้จากฟังก์ชันของเรา:
$$
\mathbf{J}_{custom} =
\begin{bmatrix}
8.9943 & -4.9379 \times 10^{-17} & -2.3356 \times 10^{-17} \\
-1.1682 \times 10^{-16} & -8.9943 & -4.7443 \\
-0.109 & -0.093 & -0.093 \\
1.2246 \times 10^{-16} & 1 & 1 \\
1 & -6.1232 \times 10^{-17} & -6.1232 \times 10^{-17} \\
-6.1232 \times 10^{-17} & 6.1232 \times 10^{-17} & 6.1232 \times 10^{-17}
\end{bmatrix}
$$

#### Jacobian ที่ได้จาก Robotic Toolbox:
$$
\mathbf{J}_{toolbox} =
\begin{bmatrix}
8.9943 & 0 & 0 \\
-5.5074 \times 10^{-17} & -8.9943 & -4.7443 \\
-0.109 & -0.093 & -0.093 \\
6.1232 \times 10^{-17} & 1 & 1 \\
1 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

### ความแตกต่างระหว่าง Jacobian ที่คำนวณได้และ Robotic Toolbox:
$$
\Delta \mathbf{J} =
\begin{bmatrix}
0 & -4.9379 \times 10^{-17} & -2.3356 \times 10^{-17} \\
-6.1749 \times 10^{-17} & 0 & 5.5511 \times 10^{-17} \\
5.5511 \times 10^{-17} & 5.5511 \times 10^{-17} & 2.7756 \times 10^{-17} \\
6.1232 \times 10^{-17} & 0 & 0 \\
0 & -6.1232 \times 10^{-17} & -6.1232 \times 10^{-17} \\
-6.1232 \times 10^{-17} & 6.1232 \times 10^{-17} & 6.1232 \times 10^{-17}
\end{bmatrix}
$$

    ```

## 2. ตรวจสอบ Singularity: checkSingularityHW3(q)
โค้ดนี้มีหน้าที่ ตรวจสอบสถานะ Singularity ของหุ่นยนต์ โดยใช้ Jacobian Matrix ซึ่งเป็นเครื่องมือสำคัญใน Kinematics เพื่อบอกว่าหุ่นยนต์สูญเสียความอิสระในการเคลื่อนที่ (Degrees of Freedom - DOF) หรือไม่

    Input:
    -**q: List ของการตั้งค่าข้อต่อ เช่น [q1, q2, q3]
    Output:
    -จะบอกว่าหุ่นยนต์อยู่ในสถานะ Singularity หรือไม่ ถ้า 0 = อยู่ในสภาวะปกติ , 1 = อยู่ในสภาวะ Singularity

    ## Singularity และ Determinant ของ Jacobian

**Singularity** คือสถานะที่หุ่นยนต์สูญเสียบางอิสระในการเคลื่อนที่ (Degrees of Freedom - DOF).  
ตัวอย่างเช่น เมื่อข้อต่อของหุ่นยนต์เรียงตัวในแนวเดียวกัน ทำให้หุ่นยนต์ไม่สามารถเคลื่อนที่ในบางทิศทางได้.

    ---

### การตรวจสอบ Singularity

การตรวจสอบ Singularity ทำได้โดย:
1. **คำนวณ Determinant ของ Jacobian Matrix (เฉพาะส่วน Translational \( J_v \))**:
   $$
   \text{det}(J_v)
   $$

2. **เกณฑ์การตัดสิน Singularity**:
   - ถ้า $$ \text{det}(J_v) \approx 0 $$ หุ่นยนต์จะอยู่ในสถานะ **Singularity**.
   - ถ้า $$ \text{det}(J_v) \neq 0 $$ หุ่นยนต์สามารถเคลื่อนที่ได้อย่างอิสระ.

    ---

โค้ดการตรวจสอบ Singularity ใน Python

```bash

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

    ```
    คำอธิบาย :
    -**การคำนวณ Reduced Jacobian ใช้เฉพาะ ส่วน Translational (3x3) ของ Jacobian Matrix ซึ่งเป็นส่วนที่ใช้คำนวณตำแหน่งของปลายหุ่นยนต์.
    -**หากค่า Determinant ของ Reduced Jacobian ใกล้ 0 (น้อยกว่าค่า epsilon), หุ่นยนต์จะถือว่าอยู่ใน Singularity.
    ```
    Test :

    ```bash
    # ตัวอย่างการทดสอบฟังก์ชัน
    q_initial = [0.0, 0.0, 0.0]  # Configuration เริ่มต้น
    flag = checkSingularityHW3(q_initial)
    print(f"Singularity Flag (q_initial): {flag}")

    q_singularity = [0.0, 0.0, 3]  # Configuration ที่อาจเป็น Singularity
    flag = checkSingularityHW3(q_singularity)
    print(f"Singularity Flag (q_singularity): {flag}")
    ```
    ได้ผลลัพธ์ดังนี้ :
        Determinant ของ Reduced Jacobian: 0.03554997075000001
        Singularity Flag (q_initial): 0

        Determinant ของ Reduced Jacobian: 0.0006170844682311913
        Singularity Flag (q_singularity): 1

    ```
## 3.คำนวณแรงบิด (Effort): computeEffortHW3(q, w)
    Input:
    -**q: List ของมุมข้อต่อ เช่น [q1, q2, q3]
    -**w: List ของค่า Wrench [Fx, Fy, Fz, Tx, Ty, Tz]
    Output:
    -**tau: List ของแรงบิดที่ข้อต่อ เช่น [τ1, τ2, τ3]
    ```bash
    def computeEffortHW3(q:list[float], w:list[float])->list[float]:

    # เรียกใช้ฟังก์ชันที่คำนวณ Jacobian ของหุ่นยนต์
    J = endEffectorJacobianHW3(q)
    
    # คำนวณแรงบิดที่ข้อต่อ: tau = J^T * w
    tau = np.dot(J.T, w)
    
    return tau
    ```
     คำอธิบาย :
    -**คำนวณแรงบิด (𝜏) ที่ข้อต่อจาก Wrench ที่ปลายมือ.
    -**ใช้สมการ $$\tau = J^T \cdot w$$\เพื่อแปลงค่าแรงที่ปลายมือเป็นแรงบิดที่ข้อต่อ.
    ```
    Test :

    ```bash
    # ทดสอบฟังก์ชัน
    q_test = [0.0, np.pi/4, np.pi/4]  # ค่า configuration ของข้อต่อ
    w_test = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # ค่า wrench (moment และ force)

    tau_result = computeEffortHW3(q_test, w_test)
    print("Effort (Torque) at the joints:", tau_result)

    ```
    คำอธิบาย :
    -**q_test: เป็นการตั้งค่ามุมของข้อต่อหุ่นยนต์ [0.0, π/4, π/4]
    -**w_test: เป็นค่า Wrench ที่ประกอบด้วยแรง (Force) และโมเมนต์ (Moment) [1.0, 2.0, 3.0, 0.1, 0.2, 0.3].
    ```
    ได้ผลลัพธ์ดังนี้ :
    Effort (Torque) at the joints: [-0.31047962 -0.82733962 -1.12786   ]

    แรงบิดนี้แสดงถึงผลของแรงและโมเมนต์ที่กระทำต่อปลายหุ่นยนต์และถูกแปลงเป็นแรงบิดที่ข้อต่อ.
    ```