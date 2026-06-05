import cv2
import numpy as np
import time

def main():
    # ================= 配置区域 =================
    # 1. 矩阵路径 (必须是你在高清分辨率下重新标定生成的那个文件！)
    MATRIX_PATH = 'ir_to_vis_homography.npy'
    
    # 2. 摄像头设备索引
    VIS_CAMERA_ID = 1  
    IR_CAMERA_ID = 3  

    # 3. 硬件抓取分辨率 (必须与你标定时的分辨率一模一样！)
    CAPTURE_WIDTH = 1920
    CAPTURE_HEIGHT = 1080

    # 4. 融合透明度与本地预览窗口大小
    ALPHA = 0.5        
    # 为了防止 1080P 原图太大撑爆你的电脑屏幕，我们设定一个预览窗口的固定宽度
    # 高度程序会自动等比计算，保证画面绝对不变形！
    PREVIEW_TARGET_WIDTH = 1280 
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

    # ================= 核心修改：强制拉升到高清分辨率 =================
    print("正在向硬件请求高清分辨率与 MJPG 格式...")
    # 可见光设置
    cap_vis.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_vis.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap_vis.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    
    # 红外设置
    cap_ir.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_ir.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap_ir.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # 验证实际生效的分辨率
    actual_vis_w = cap_vis.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_vis_h = cap_vis.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_ir_w = cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_ir_h = cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"📺 实际生效 -> 可见光: {actual_vis_w}x{actual_vis_h} | 红外: {actual_ir_w}x{actual_ir_h}")
    # =================================================================

    print("===========================================")
    print("🚀 高清实时对齐已启动 (自动裁剪公共视野)！")
    print("操作提示：")
    print("  - 按 'q' 或 'ESC' 键退出")
    print("  - 按 'w' 键只看可见光 (Alpha=1.0)")
    print("  - 按 's' 键只看红外 (Alpha=0.0)")
    print("  - 按 'a' 键恢复半透明融合 (Alpha=0.5)")
    print("===========================================")

    cv2.namedWindow("Dual-Camera HD Alignment Live", cv2.WINDOW_AUTOSIZE)

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

        # ================= 仅计算一次裁剪框 =================
        if crop_box is None:
            gray_ir = cv2.cvtColor(aligned_ir, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_ir, 1, 255, cv2.THRESH_BINARY)
            
            cx, cy, cw, ch = cv2.boundingRect(thresh)
            
            margin = 4 # 高清下留多一点安全边距
            crop_box = (cx + margin, cy + margin, cw - margin*2, ch - margin*2)
            print(f"✂️ 已锁定高清重叠区域坐标: (X:{crop_box[0]}, Y:{crop_box[1]}, W:{crop_box[2]}, H:{crop_box[3]})")
        # ====================================================

        # 从原图和对齐图中裁剪出纯重叠部分
        cx, cy, cw, ch = crop_box
        cropped_vis = frame_vis[cy:cy+ch, cx:cx+cw]
        cropped_ir = aligned_ir[cy:cy+ch, cx:cx+cw]

        # 融合裁剪后的图像
        blended = cv2.addWeighted(cropped_vis, ALPHA, cropped_ir, 1 - ALPHA, 0)

        # ================= 核心修改：等比例缩放预览 =================
        # 1. 计算裁剪后画面的真实高宽比
        aspect_ratio = ch / cw  
        # 2. 根据你设定的预览宽度，算出不会变形的预览高度
        preview_height = int(PREVIEW_TARGET_WIDTH * aspect_ratio)
        # 3. 缩放显示 (注意：这只是为了屏幕显示，如果你后续要把 blended 喂给神经网络，直接用 blended 原图！)
        final_display = cv2.resize(blended, (PREVIEW_TARGET_WIDTH, preview_height))
        # ============================================================

        # 计算并显示 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(final_display, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(final_display, f"Vis Alpha: {ALPHA:.2f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 显示画面
        cv2.imshow("Dual-Camera HD Alignment Live", final_display)

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