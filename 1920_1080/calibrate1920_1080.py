import cv2
import numpy as np
import math

# ================= 全局变量 =================
image_pairs = []       # 存放所有抓拍的图像对 [{'vis': img, 'ir': img}, ...]
all_pts_ir = []        # 存放所有图片的 IR 点
all_pts_vis = []       # 存放所有图片的 VIS 点

# 当前正在处理的图片的特征点
curr_pts_ir = []
curr_pts_vis = []

# 交互状态
active_idx = -1        # 当前激活的点索引
active_cam = None      # 当前激活的窗口 ('ir' 或 'vis')
img_ir_base = None     # 当前处理的 IR 底图
img_vis_base = None    # 当前处理的 VIS 底图

# ================= 辅助函数 =================
def get_closest_point(x, y, pts, threshold=10):
    """寻找距离点击位置最近的特征点，用于重新激活"""
    if not pts:
        return -1
    dists = [math.hypot(px - x, py - y) for px, py in pts]
    min_idx = np.argmin(dists)
    if dists[min_idx] < threshold:
        return min_idx
    return -1

def update_display():
    """刷新显示窗口，绘制特征点和激活状态的准星"""
    global img_ir_base, img_vis_base, curr_pts_ir, curr_pts_vis, active_idx, active_cam
    
    if img_ir_base is None or img_vis_base is None:
        return

    disp_ir = img_ir_base.copy()
    disp_vis = img_vis_base.copy()

    # 绘制 IR 点
    for i, pt in enumerate(curr_pts_ir):
        is_active = (active_cam == 'ir' and i == active_idx)
        color = (0, 255, 255) if is_active else (0, 0, 255) # 激活为黄色，平时为红色
        cv2.circle(disp_ir, tuple(pt), 4, color, -1)
        cv2.putText(disp_ir, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_active: # 画准星
            cv2.line(disp_ir, (pt[0]-10, pt[1]), (pt[0]+10, pt[1]), color, 1)
            cv2.line(disp_ir, (pt[0], pt[1]-10), (pt[0], pt[1]+10), color, 1)

    # 绘制 VIS 点
    for i, pt in enumerate(curr_pts_vis):
        is_active = (active_cam == 'vis' and i == active_idx)
        color = (0, 255, 255) if is_active else (0, 255, 0) # 激活为黄色，平时为绿色
        cv2.circle(disp_vis, tuple(pt), 4, color, -1)
        cv2.putText(disp_vis, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_active: # 画准星
            cv2.line(disp_vis, (pt[0]-10, pt[1]), (pt[0]+10, pt[1]), color, 1)
            cv2.line(disp_vis, (pt[0], pt[1]-10), (pt[0], pt[1]+10), color, 1)

    # 显示操作提示
    tips = "WASD: Fine-tune | Z: Undo | N: Next Image/Compute"
    cv2.putText(disp_ir, tips, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(disp_vis, tips, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Calibration - IR", disp_ir)
    cv2.imshow("Calibration - VIS", disp_vis)

# ================= 鼠标回调函数 =================
def click_ir(event, x, y, flags, param):
    global curr_pts_ir, active_idx, active_cam
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = get_closest_point(x, y, curr_pts_ir)
        if idx != -1:
            active_idx = idx # 点击靠近旧点，激活旧点
        else:
            curr_pts_ir.append([x, y]) # 添加新点
            active_idx = len(curr_pts_ir) - 1
        active_cam = 'ir'
        update_display()

def click_vis(event, x, y, flags, param):
    global curr_pts_vis, active_idx, active_cam
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = get_closest_point(x, y, curr_pts_vis)
        if idx != -1:
            active_idx = idx
        else:
            curr_pts_vis.append([x, y])
            active_idx = len(curr_pts_vis) - 1
        active_cam = 'vis'
        update_display()

# ================= 主程序 =================
def main():
    global image_pairs, all_pts_ir, all_pts_vis
    global curr_pts_ir, curr_pts_vis, active_idx, active_cam, img_ir_base, img_vis_base

    VIS_CAMERA_ID = 1
    IR_CAMERA_ID = 3

    cap_vis = cv2.VideoCapture(VIS_CAMERA_ID)
    cap_ir = cv2.VideoCapture(IR_CAMERA_ID)

    if not cap_vis.isOpened() or not cap_ir.isOpened():
        print("❌ 无法同时打开两个摄像头，请检查连接或设备ID。")
        return
    # ================= 新增：强制修改为高清分辨率 =================
    # 强烈建议强制开启 MJPG 格式，否则两个高清摄像头同时开极大概率 USB 带宽超载卡死
    cap_vis.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_vis.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 或者 1280
    cap_vis.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 或者 720
    
    cap_ir.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_ir.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
    cap_ir.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
    
    # 打印实际生效的分辨率，让你心里有底
    print(f"可见光实际分辨率: {cap_vis.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_vis.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"红外实际分辨率: {cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    # ==============================================================
    
    print("===========================================")
    print("📸 第一阶段：多图同步抓拍")
    print("将双光摄像头对准一面【平整且有丰富纹理】的墙面/标定板。")
    print("- 按 'SPACE' 抓拍当前画面（可多次抓拍不同视角的平整墙面）。")
    print("- 按 'C' 结束抓拍，进入标定环节。")
    print("- 按 'Q' 退出。")
    print("===========================================")

    # 1. 抓拍阶段
    while True:
        ret_vis, vis_img = cap_vis.read()
        ret_ir, ir_img = cap_ir.read()

        if not ret_vis or not ret_ir:
            print("读取视频流失败！")
            break

        vis_show = cv2.resize(vis_img, (vis_img.shape[1]//2, vis_img.shape[0]//2))
        ir_show = cv2.resize(ir_img, (ir_img.shape[1]//2, ir_img.shape[0]//2))
        preview = np.hstack((vis_show, ir_show))
        
        # 显示已抓拍数量
        cv2.putText(preview, f"Captured Pairs: {len(image_pairs)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Live Preview (SPACE: Capture, C: Calibrate)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap_vis.release()
            cap_ir.release()
            cv2.destroyAllWindows()
            return
        elif key == 32:  # 空格键
            image_pairs.append({'vis': vis_img.copy(), 'ir': ir_img.copy()})
            print(f"✅ 成功抓拍第 {len(image_pairs)} 对图像！")
        elif key == ord('c'):
            if len(image_pairs) == 0:
                print("⚠️ 至少需要抓拍1对图像！")
            else:
                print(f"进入标定模式，共计 {len(image_pairs)} 对图像...")
                break

    cap_vis.release()
    cap_ir.release()
    cv2.destroyAllWindows()

    # 2. 标定阶段 (遍历多图)
    cv2.namedWindow("Calibration - IR", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Calibration - VIS", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration - IR", click_ir)
    cv2.setMouseCallback("Calibration - VIS", click_vis)

    print("\n===========================================")
    print("🎯 第二阶段：精细特征点标定")
    print("- 鼠标【左键】添加特征点。")
    print("- 鼠标点击【已有特征点】可重新激活它。")
    print("- 键盘【W/A/S/D】对激活的点(黄十字)进行像素级微调。")
    print("- 键盘【Z】删除当前激活的点。")
    print("- 键盘【N】确认当前图片的选点，进入下一张（或计算矩阵）。")
    print("===========================================")

    for pair_idx, pair in enumerate(image_pairs):
        img_ir_base = pair['ir']
        img_vis_base = pair['vis']
        curr_pts_ir = []
        curr_pts_vis = []
        active_idx = -1
        active_cam = None

        print(f"\n---> 正在标注第 {pair_idx + 1} / {len(image_pairs)} 对图像")
        update_display()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27: # ESC 直接退出整个程序
                cv2.destroyAllWindows()
                return
            
            # WASD 微调逻辑
            if key in [ord('w'), ord('a'), ord('s'), ord('d')] and active_idx != -1:
                target_pts = curr_pts_ir if active_cam == 'ir' else curr_pts_vis
                if active_idx < len(target_pts):
                    if key == ord('w'): target_pts[active_idx][1] -= 1
                    elif key == ord('s'): target_pts[active_idx][1] += 1
                    elif key == ord('a'): target_pts[active_idx][0] -= 1
                    elif key == ord('d'): target_pts[active_idx][0] += 1
                    update_display()
            
            # Z 撤销逻辑
            elif key == ord('z') and active_idx != -1:
                target_pts = curr_pts_ir if active_cam == 'ir' else curr_pts_vis
                if active_idx < len(target_pts):
                    target_pts.pop(active_idx)
                    active_idx = -1
                    update_display()

            # N 下一步逻辑
            elif key == ord('n'):
                if len(curr_pts_ir) != len(curr_pts_vis):
                    print(f"⚠️ 当前图像点数不匹配！IR: {len(curr_pts_ir)}, VIS: {len(curr_pts_vis)}")
                    continue
                # 将当前图片的点汇总到全局列表
                all_pts_ir.extend(curr_pts_ir)
                all_pts_vis.extend(curr_pts_vis)
                print(f"✅ 第 {pair_idx + 1} 对图像已确认，贡献了 {len(curr_pts_ir)} 对点。")
                break 

    cv2.destroyWindow("Calibration - IR")
    cv2.destroyWindow("Calibration - VIS")

    # 3. 计算与融合预览
    if len(all_pts_ir) < 4:
        print("\n❌ 错误：所有图像加起来的点数不足 4 对，无法计算单应性矩阵。")
        return

    print(f"\n===========================================")
    print(f"⚙️ 第三阶段：计算矩阵 (共采用 {len(all_pts_ir)} 对特征点)")
    
    src_pts = np.array(all_pts_ir, dtype=np.float32)
    dst_pts = np.array(all_pts_vis, dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        # 取第一对图像做预览
        test_vis = image_pairs[0]['vis']
        test_ir = image_pairs[0]['ir']
        h, w = test_vis.shape[:2]
        aligned_ir = cv2.warpPerspective(test_ir, H, (w, h))

        blended = cv2.addWeighted(test_vis, 0.5, aligned_ir, 0.5, 0)
        
        cv2.namedWindow("Final Alignment Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Alignment Preview", blended)
        print("预览已生成（使用第一对图像）。按 'S' 保存矩阵，按 'Q' 退出。")
        
        while True:
            b_key = cv2.waitKey(0) & 0xFF
            if b_key == ord('s'):
                np.save('ir_to_vis_homography.npy', H)
                print("✅ 矩阵已成功保存至 'ir_to_vis_homography.npy'！")
                break
            elif b_key == ord('q') or b_key == 27:
                break
    else:
        print("❌ 矩阵计算失败！请检查选点是否正确。")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()