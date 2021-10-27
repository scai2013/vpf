# -*- coding:utf-8 -*-
import math
import threading
import time
import traceback
from threading import Thread

from events_thread.RTSCapture import RTSCapture
from tools.get_best_gpu_id import get_used_minimum_gpu_index
from torch_utils.datasets import *
from events_thread.if1 import if1
from tools.common_utils import *
from tools.sqllite_sql import delete_camera_server_host_by_camera_id, update_camera_server_host
from utils.torch_utils import time_sync

offline_img_url = get_camera_offline_img_url()           # 视频离线消息图片URL
need_sleep = get_item_value("FREQUENCY", "NEED_SLEEP")
SEND_OFFLINE_INTERVAL = int(get_item_value('FREQUENCY', 'SEND_OFFLINE_INTERVAL'))
need_sleep = True if (need_sleep is not None and need_sleep == 'TRUE') else False
logger.info('need_sleep is %s' % str(need_sleep))
skip_number = get_item_value("FREQUENCY", "SKIP_NUMBER")
OFFTIME_LIMIT = int(get_item_value('VIDEO_PROCESS', 'OFFTIME_LIMIT'))

skip_number = int(skip_number) if skip_number is not None else 3
buffer_size = int(get_item_value("DECODE", "CAP_PROP_BUFFERSIZE"))
buffer_size = int(buffer_size) if buffer_size is not None else 3
ifo_default_size = int(get_item_value('MEMORY', 'IFO_SIZE'))
'''
    function: VpfYolov5sDetector 车辆目标检测主类
    author:daizhisheng
    date:2020-04-23
'''

if os.name == 'nt':
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(';')
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import pycuda.driver as cuda
import PyNvCodec as nvc
import numpy as np
from threading import Thread

class VpfYolov5sDetector:
    """
    "
    """
    instance = {}
    instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls in cls.instance:
            return cls.instance[cls]['obj']

        cls.instance_lock.acquire()
        try:
            if cls not in cls.instance:
                obj = object.__new__(cls)
                cls.instance[cls] = {'obj': obj, 'init': False}
                setattr(cls, '__init__', cls.__decorate_init(cls.__init__))
        finally:
            pass
        cls.instance_lock.release()
        return cls.instance[cls]['obj']

    @classmethod
    def __decorate_init(cls, fn):
        def init_wrap(*args):
            if not cls.instance[cls]['init']:
                fn(*args)
                cls.instance[cls]['init'] = True
            return
        return init_wrap

    def __init__(self, max_thread_num=8, img_size=416, stride=32):
        logger.info("PytorchYolov5sDetector 初始化...")
        self.img_size = img_size
        self.stride = stride
        self.last_img_list = [None] * max_thread_num
        self.last_mask_roi_list = [None] * max_thread_num
        self.last_roi_list = [None] * max_thread_num
        self.last_frame_num_list = [None] * max_thread_num
        self.sources = [None] * max_thread_num
        self.detect_area = [None] * max_thread_num
        self.current_index = 0
        self.last_time = None

    def select_free_location(self):
        for i, box in enumerate(self.sources):
            if box is None:
                return i
        return -1

    def free_location(self, index):
        self.sources[index] = None
        self.last_mask_roi_list[index] = None
        self.last_roi_list[index] = None
        self.last_img_list[index] = None
        self.detect_area[index] = None
        self.last_frame_num_list[index] = None
        self.current_index -= 1

    def video_init(self, task_paras_json, queue=None, mode=1):
        index = self.select_free_location()
        if index == -1:
            logger.error("该容器无法再创建视频分析服务！")
            raise Exception("该容器无法再创建视频分析服务！")
            return
        camera_common_info = task_paras_json.get("camera_common_info", {})  # 摄像头基础信息
        detect_area_pool = task_paras_json.get("detect_area_pool", {})  # 检测区域池
        events_json = task_paras_json.get("events", {})  # 事件集
        assert events_json is not None and len(events_json) > 0, "没有配置需要检测的事件！"
        s_camera_id = camera_common_info.get("s_camera_id", None)  # 摄像头控编号
        input_stream = camera_common_info.get('s_video_url', None)

        logger.info(
            "s_camera_id = " + str(s_camera_id) + ",input_stream=" + str(input_stream) + ",detect_area_pool="
            + str(detect_area_pool))

        # video_quality_analysis_event = events_json.get("video_quality_analysis_event", {})  # 视频质量事件配置
        # video_quality_switch = video_quality_analysis_event.get("switch", 0)  # 视频质量开关
        # logger.info("视频质量分析开关：" + s tr(video_quality_switch))

        check_area = None
        check_area_key = None  # 检测区域的Key
        if len(detect_area_pool) > 0:  # 假设只有一个检测区域
            for k, v in detect_area_pool.items():
                check_area_key = k  # 用的是检测区域池中的哪个区域名称
                check_area = v.get("position")  # 多边形检测区域（四边形）

        if not all([s_camera_id, input_stream, check_area]):
            logger.error("输入视频源参数或检测区域为空,停止分析！")
            return

        thread = Thread(target=self.video_process,
                        args=([index, s_camera_id, input_stream, task_paras_json, check_area_key]), daemon=True)
        thread.start()
        self.current_index += 1
        if1.initIF0(s_camera_id)

    def video_process(self, index, s_camera_id, input_stream, task_paras_json, check_area_key):
        """
            function: 视频分析（车辆检测主程序）
            input:
                task_paras_json:     摄像头检测参数（来自配置页面）
            author: daizhisheng
            date:2020-11-9
        """
        gpuID = int(get_used_minimum_gpu_index())
        total_time = 0
        nvDec = None
        nvCvt = None
        nvRes = None
        nvDwn = None
        option = {'stream_loop': '-1', 'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'}
        try:
            logger.info('开始处理视频流: ' + str(s_camera_id))
            start_time = time.time()
            camera_alive_status_dict.update({s_camera_id: 1})
            logger.info('video_stream %s initialized correct' % input_stream)

            ctx = cuda.Device(gpuID).retain_primary_context()
            ctx.push()
            nv_str = cuda.Stream()
            ctx.pop()

            # Create Decoder with given CUDA context & stream.
            nvDec = nvc.PyNvDecoder(input_stream, gpuID, option)

            initial_w, initial_h = nvDec.Width(), nvDec.Height()
            # hwidth, hheight = int(initial_w / 2), int(initial_h / 2)
            # hwidth = hwidth
            # hheight = hheight
            fps = nvDec.Framerate()


            # Determine colorspace conversion parameters.
            # Some video streams don't specify these parameters so default values
            # are most widespread bt601 and mpeg.
            cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
            if nvc.ColorSpace.UNSPEC == cspace:
                cspace = nvc.ColorSpace.BT_601
            if nvc.ColorRange.UDEF == crange:
                crange = nvc.ColorRange.MPEG
            cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
            logger.info(s_camera_id + "摄像头视频分辨率：" + str(initial_w) + " x "+str(initial_h)
                        + ",fps="+str(fps) + ",Color space=" + str(cspace) +",Color range=" + str(crange))

            nvCvt = nvc.PySurfaceConverter(initial_w, initial_h, nvDec.Format(), nvc.PixelFormat.RGB, ctx.handle, nv_str.handle)
            nvRes = nvc.PySurfaceResizer(initial_w, initial_h, nvCvt.Format(), ctx.handle, nv_str.handle)
            nvDwn = nvc.PySurfaceDownloader(initial_w, initial_h, nvRes.Format(), ctx.handle, nv_str.handle)

            detect_area_pool = task_paras_json.get("detect_area_pool", {})  # 检测区域池
            assert detect_area_pool is not None and len(detect_area_pool) > 0, "IF0 init error,detect_area_pool为空！"
            check_area = detect_area_pool.get(check_area_key, {}).get("position")
            assert check_area is not None, \
                "IF0 init error,检测区域池中check_area的position为空！check_area_key="+str(check_area_key)
            smallest_external_rectangle = get_polygon_smallest_external_rectangle(check_area)  # 检测区域最小外接矩形框
            logger.info("页面传入多边形坐标：check_area = " + str(check_area))
            logger.info("多边形的最小外接矩形：smallest_external_rectangle" + str(smallest_external_rectangle))

            # 计算check_area基于smallest_external_rectangle左上角的相对坐标,用来在check_area的ROI矩形区域内生成只有路面检测的区域。
            new_check_area = cal_polygon_relative_position(check_area, smallest_external_rectangle)
            roi_left = smallest_external_rectangle[0]
            roi_right = smallest_external_rectangle[2]
            roi_top = smallest_external_rectangle[1]
            roi_bottom = smallest_external_rectangle[3]
            detect_area_square = (roi_right - roi_left) * (roi_bottom - roi_top)  # 检测区域面积

            current_photo_id = 0
            num_frame = 0
            frame_interval_time = 1/fps
            failure_time = 0
            beg_time = time.time()
            rawSurface = None
            while True:
                try:
                    rawSurface = nvDec.DecodeSingleSurface()
                    if (rawSurface.Empty()):
                        logger.error('No more video frames')
                        failure_time += 1
                        if failure_time > 3:
                            nvDec = nvc.PyNvDecoder(input_stream, gpuID, option)
                            nvCvt = nvc.PySurfaceConverter(initial_w, initial_h, nvDec.Format(), nvc.PixelFormat.RGB, ctx.handle, nv_str.handle)
                            nvRes = nvc.PySurfaceResizer(initial_w, initial_h, nvCvt.Format(), ctx.handle, nv_str.handle)
                            nvDwn = nvc.PySurfaceDownloader(initial_w, initial_h, nvRes.Format(), ctx.handle, nv_str.handle)
                        time.sleep(3)
                        continue
                except nvc.HwResetException:
                    traceback.print_stack()
                    logger.error('Continue after HW decoder was reset')
                    continue

                num_frame += 1
                if num_frame % skip_number != 0:
                    time.sleep(frame_interval_time)
                    continue
                cvtSurface = nvCvt.Execute(rawSurface, cc_ctx)
                if (cvtSurface.Empty()):
                    logger.error('Failed to dod color conversion')
                    continue

                resSurface = nvRes.Execute(cvtSurface)
                if (resSurface.Empty()):
                    logger.error('Failed to resize surface')
                    continue

                rawFrame = np.ndarray(shape=(resSurface.HostSize()), dtype=np.uint8)
                success = nvDwn.DownloadSingleSurface(resSurface, rawFrame)
                if not (success):
                    logger.error('Failed to download surface')
                    continue
                bgr = rawFrame.reshape(initial_h, initial_w, 3)
                frame = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
                # if s_camera_id == 0:
                #     cv2.imshow(str(s_camera_id), img)
                #     cv2.waitKey(1)
                current_photo_id += 1
                roi_image = frame[roi_top:roi_bottom, roi_left:roi_right]  # 提取检测区域
                mask_roi_image = set_ROI_mask(roi_image, np.array(new_check_area))
                self.last_mask_roi_list[index] = letterbox(mask_roi_image, self.img_size, auto=False)[0]
                self.last_roi_list[index] = roi_image
                self.last_img_list[index] = frame
                self.last_frame_num_list[index] = num_frame
                self.sources[index] = s_camera_id
                self.detect_area[index] = detect_area_square

                if current_photo_id % status_update_fps == 0:
                    update_camera_server_host(s_camera_id, current_server_host)  # 更新表数据
                    current_photo_id = 0

                status = camera_alive_status_dict.get(s_camera_id)
                if status is None or status == 0:
                    self.free_location(index)
                    logger.warning('free_location %s' %(str(index)))
                    logger.warning(s_camera_id + " task break! camera alive status = " + str(status))
                    break

                wait_time = frame_interval_time - (time.time() - beg_time)
                if wait_time > 0:
                    logger.warning("wait_timewait_timewait_timewait_time sleep: " + str(wait_time))
                    time.sleep(wait_time)

            logger.warning("video analysis terminated, s_camera_id=" + str(s_camera_id))
            end_time = time.time()
            logger.info('video task {} process frames {} using time {:.2f} s '.format(s_camera_id, num_frame, end_time - start_time))

        except Exception as e:
            traceback.print_stack()
            logger.error(e, exc_info=True)
            logger.error(traceback.format_exc())
        finally:
            delete_camera_server_host_by_camera_id(s_camera_id)
            self.free_location(index)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        sources = self.sources.copy()

        sources = [x for x in sources if x is not None]

        sources_len = len(sources)
        if sources_len == 0:
            return None, None, None, None, None, None
        self.count += 1

        img1 = self.last_img_list.copy()
        img1 = [x for x in img1 if x is not None]
        img1 = img1[0:sources_len]

        img0 = self.last_roi_list.copy()
        img0 = [x for x in img0 if x is not None]
        img0 = img0[0:sources_len]

        imgs = self.last_mask_roi_list.copy()
        imgs = [x for x in imgs if x is not None]
        imgs = imgs[0:sources_len]

        try:
            # img = [letterbox(x, self.img_size, auto=False)[0] for x in imgs if x is not None]
            img = np.stack(imgs, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
            img = np.ascontiguousarray(img)
        except Exception as e:
            traceback.print_stack()
            logger.error(e)
            return None, None, None, None, None

        detect_area = self.detect_area.copy()
        detect_area = [x for x in detect_area if x is not None]
        detect_area = detect_area[0:sources_len]

        frame_num_list = self.last_frame_num_list.copy()
        frame_num_list = [x for x in frame_num_list if x is not None]
        frame_num_list = frame_num_list[0:sources_len]

        return sources, img, img0, img1, detect_area, frame_num_list

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

vpfYolov5sDetector = VpfYolov5sDetector(8, 416, 32)