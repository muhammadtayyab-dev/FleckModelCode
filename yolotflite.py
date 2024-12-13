import glob
import os

import cv2
from torch import Tensor
from typing import Tuple
from ultralytics.yolo.utils import ops
import torch
from ultralytics.yolo.engine.results import Results
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def letterbox(img: np.ndarray,
            new_shape:Tuple[int, int] = (640, 640),
            color:Tuple[int, int, int] = (114, 114, 114),
            auto:bool = False, 
            scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
    
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride

    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size

    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    print(np.shape(img))
    # img = img.transpose(2, 0, 1)
    # print(np.shape(img))
    img = np.ascontiguousarray(img)
    # print(np.shape(img))

    return img

def overlay(image, mask, box, clas, color, alpha, resize=None):

    color = color[::-1]
    # image = image[0]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    cv2.imwrite("masked_.png",colored_mask)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    cv2.imwrite("masked2.png",colored_mask)
    masked = np.ma.MaskedArray(np.array(image), mask=(colored_mask), fill_value=color)

    cv2.imwrite("testMasked.png",masked)
    image_overlay = masked.filled()
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    cv2.imwrite("my.png",image_overlay)
    # test = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 234), 2)
    # test = image[row:(int(box[0])+int(box[1])),column:column+width]
    # cv2.imwrite("my.png",test)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    image_combined = cv2.rectangle(image_overlay, (int(box[0]), int(box[1])), (int(box[2])+20, int(box[3])+20), (255, 0, 234), 5)
    image_combined = cv2.putText(image_overlay, str("Tayyab"), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_DUPLEX,4, (0, 0, 0), 10)
    cv2.imwrite("my2.png",image)
    return image_combined

def image_to_tensor(image:np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


import time

def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
):
   
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
 
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence
        #shape of x here is 29:40]
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        # best class only
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
    return output


def postprocess(path, preds, img, orig_imgs, proto):
    """
      Post-processes object detection predictions and returns the results.

      Args:
      path: A string or list of strings representing the path(s) of the image(s) being processed.
      preds: A tensor containing the predictions for the image(s).
      img: A tensor representing the image(s) being processed.
      orig_imgs: A tensor or list of tensors representing the original image(s).
      proto: A tensor representing the proto output.

      Returns:
      A list of Results objects, containing the post-processed predictions.
      """
    # Perform non-maximum suppression on the predictions to remove overlapping boxes.
    p = non_max_suppression(preds,
                                    0.15,
                                    0.5,
                                    nc=4,
                                # agnostic=True
                                    )
    print("End Time : ",datetime.datetime.now())
   
    retina_masks = False #
    results = []
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        print("Detected Image : ",pred.shape)
        # path, _, _, _, _ = self.batch
        path = path
        img_path = path[i] if isinstance(path, list) else path
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names=['a','b','c','d'], boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[1:3], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)


        results.append(
            Results(orig_img=orig_img, path=img_path, names=['a','b','c','d'], boxes=pred[:, :6], masks=masks))
        print(i)
    return results

from PIL import Image
#FingerPrint_models/Tflite_Model/
model_path = "best_float16.tflite"
image_paths  =  glob.glob('HandandFP04Redmi_10/*.jpg') # You can also add other extensions
output_path = 'lite16_outputs'

# Create Output Directory if it doesnot exist
os.makedirs(output_path,exist_ok=True)

interpreter = tf.lite.Interpreter(model_path=model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
print()
print("Input details:")
print(input_details) # The Model Input Specifications
print()
print("Output details:")
print(output_details)
print()
import datetime
for IMAGE_PATH in image_paths:
    # IMAGE_PATH = '/Users/rizwan/Ihsan_FingerPrint_Project/Labelling update on 13th August/Data_for_Yolo_Finger/train/images/IMG (183).jpg'
    input_image = np.array(Image.open(IMAGE_PATH))
    print("Start Time : ",datetime.datetime.now())
    fname = IMAGE_PATH.split('/')[-1]
    preprocessed_image1 = preprocess_image(input_image)
    preprocessed_image = image_to_tensor(preprocessed_image1)

    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)

    # run the inference
    interpreter.invoke()
    # grab the outputs
    output1 = interpreter.get_tensor(output_details[1]['index'])  # Mask protos
    
    output1 = np.moveaxis(output1, -1, 1) # Rearrange the dimensions in order to be accepted for post processing

    output2 = interpreter.get_tensor(output_details[0]['index'])  # [batch, 4 + num_classes + num_masks, num_predictions]
    # Apply Postprocessing on the results
    print(output2[0][0][0])
    result = postprocess(
        IMAGE_PATH,
        torch.from_numpy(output2),
        preprocessed_image,
        input_image, 
        torch.from_numpy(output1))

    cls = result[0].boxes.cls.cpu().numpy()  # Class Label, (N, 1) # N is batch size, here it is 1

    # Display the results on the preprocessed image
    image_with_masks = np.copy(preprocessed_image1) # If you want to overlay on original image, then select the original image here
    masks = result[0].masks.masks.cpu().numpy()  # masks, (N, H, W)
    boxes = result[0].boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)

    for box, mask_i, ci in zip(boxes, masks, cls):
        image_with_masks = overlay(image_with_masks, mask_i, box, ci, color=(0,0,255), alpha=0.3)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image_with_masks.shape[0],image_with_masks.shape[1])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_with_masks, aspect=1)
    fig.savefig(os.path.join(output_path,fname), dpi=35)
