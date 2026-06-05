import cv2
import numpy as np
import time

def main():
    # ================= 配置区域 =================
    # 1. 矩阵路径
    MATRIX_PATH = 'ir_to_vis_homography.npy'
    
    # 2. 摄像头设备索引
    VIS_CAMERA_ID = 1  
    IR_CAMERA_ID = 3  

    # 3. 融合透明度与固定输出尺寸
    ALPHA = 0.5        
    FIXED_OUTPUT_SIZE = (1024, 768)  # 设定最终展示窗口的固定大小 (宽, 高)
    # ============================================

    # 尝试加载对齐矩阵
    try:
        H = np.load(MATRIX_PATH)
        print(f"✅ 成功加载对齐矩阵: {MATRIX_PATH}")
    except FileNotFoundError:
        print(f"❌ 找不到文件 {MATRIX_PATH}，请先运行标定脚本。")
        return

    # 初始化摄像头
    print(f"正在打开可见光摄像头 (ID: {VIS_CAMERA_ID})...")
    cap_vis = cv2.VideoCapture(VIS_CAMERA_ID)
    
    print(f"正在打开红外摄像头 (ID: {IR_CAMERA_ID})...")
    cap_ir = cv2.VideoCapture(IR_CAMERA_ID)

    if not cap_vis.isOpened() or not cap_ir.isOpened():
        print("❌ 摄像头打开失败，请检查设备连接或索引是否正确！")
        return

    print("===========================================")
    print("🚀 实时对齐已启动 (自动裁剪公共视野)！")
    print("操作提示：")
    print("  - 按 'q' 或 'ESC' 键退出")
    print("  - 按 'w' 键只看可见光 (Alpha=1.0)")
    print("  - 按 's' 键只看红外 (Alpha=0.0)")
    print("  - 按 'a' 键恢复半透明融合 (Alpha=0.5)")
    print("===========================================")

    cv2.namedWindow("Dual-Camera Alignment Live", cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    crop_box = None  # 用于存储自动计算出的公共视野边界框

    while True:
        ret_vis, frame_vis = cap_vis.read()
        ret_ir, frame_ir = cap_ir.read()

        if not ret_vis or not ret_ir:
            print("⚠️ 无法获取视频帧，视频流已中断。")
            break

        h, w = frame_vis.shape[:2]

        # 执行单应性透视变换
        aligned_ir = cv2.warpPerspective(
            frame_ir, 
            H, 
            (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0)
        )

        
        if crop_box is None:
            # 将变换后的红外图像转为灰度图，非黑色的区域即为有效重叠区域
            gray_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_ir, 1, 255, cv2.THRESH_BINARY)
            
            # 找到有效区域的最小外接矩形
            cx, cy, cw, ch = cv2.boundingRect(thresh)
            
            # 留 2 个像素的安全边距，防止边缘出现细微的黑色黑边
            margin = 2
            crop_box = (cx + margin, cy + margin, cw - margin*2, ch - margin*2)
            print(f"✂️ 已自动锁定重叠区域坐标: (X:{crop_box[0]}, Y:{crop_box[1]}, W:{crop_box[2]}, H:{crop_box[3]})")
        # ==========================================================

        # 从原图和对齐图中裁剪出纯重叠部分
        cx, cy, cw, ch = crop_box
        cropped_vis = frame_vis[cy:cy+ch, cx:cx+cw]
        cropped_ir = aligned_ir[cy:cy+ch, cx:cx+cw]

        # 融合裁剪后的图像
        blended = cv2.addWeighted(cropped_vis, ALPHA, cropped_ir, 1 - ALPHA, 0)

        # 强行缩放至用户设定的固定大小
        final_display = cv2.resize(blended, FIXED_OUTPUT_SIZE)

        # 计算并显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(final_display, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(final_display, f"Vis Alpha: {ALPHA:.2f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 显示画面
        cv2.imshow("Dual-Camera Alignment Live", final_display)

        # 键盘事件监听
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('w'):
            ALPHA = 1.0 
        elif key == ord('s'):
            ALPHA = 0.0 
        elif key == ord('a'):
            ALPHA = 0.5

    cap_vis.release()
    cap_ir.release()
    cv2.destroyAllWindows()
    print("程序已安全退出。")

if __name__ == "__main__":
    main()