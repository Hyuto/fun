import cv2
import numpy as np

import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort

from onnxsim import simplify


@gs.Graph.register()
def slice(self, data, start, end, axis=2):
    return self.layer(op="Slice", inputs=[data, start, end, axis], outputs=["slice_out_gs"])[0]


@gs.Graph.register()
def cast(self, data, to):
    return self.layer(
        op="Cast",
        inputs=[data],
        outputs=["cast_out_gs"],
        attrs={"to": to},
    )[0]


@gs.Graph.register()
def gather(self, data, indices, axis=2):
    return self.layer(
        op="Gather",
        inputs=[data, indices],
        outputs=["gather_out_gs"],
        attrs={"axis": axis},
    )[0]


@gs.Graph.register()
def transpose(self, data, axis):
    return self.layer(
        op="Transpose",
        inputs=[data],
        outputs=["transpose_out_gs"],
        attrs={"perm": axis},
    )[0]


@gs.Graph.register()
def non_max_suppression(
    self,
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    center_point_box=1,  # for yolo pytorch model
):
    # Docs : https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
    return self.layer(
        op="NonMaxSuppression",
        inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        outputs=["nms_out_gs"],
        attrs={"center_point_box": center_point_box},
    )[0]


if __name__ == "__main__":
    graph = gs.Graph()

    input_ = gs.Variable(name="detection", dtype=np.float32, shape=(1, None, None))
    config = gs.Variable(name="config", dtype=np.float32, shape=(3,))

    boxes = graph.slice(
        input_,
        start=np.asarray([0], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice boxes from outputs
    boxes.name = "raw-boxes"
    boxes.dtype = np.float32
    boxes.shape = [1, None, 4]

    confidences = graph.slice(
        input_,
        start=np.asarray([4], dtype=np.int32),
        end=np.asarray([5], dtype=np.int32),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice confidences from outputs
    confidences.name = "raw-confidences"
    confidences.dtype = np.float32
    confidences.shape = [1, None, 1]

    topk = graph.cast(
        graph.slice(
            config,
            start=np.asarray([0], dtype=np.int32),
            end=np.asarray([1], dtype=np.int32),
            axis=np.asarray([0], dtype=np.int32),
        ),  # slice topk from outputs
        to=7,
    )  # cast to int64
    topk.name = "topk"
    topk.dtype = np.int64
    topk.shape = [1]

    iou_tresh = graph.slice(
        config,
        start=np.asarray([1], dtype=np.int32),
        end=np.asarray([2], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )  # slice iou_tresh from outputs
    iou_tresh.name = "iou_tresh"
    iou_tresh.dtype = np.float32
    iou_tresh.shape = [1]

    conf_tresh = graph.slice(
        config,
        start=np.asarray([2], dtype=np.int32),
        end=np.asarray([3], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )  # slice conf_tresh from outputs
    conf_tresh.name = "conf_tresh"
    conf_tresh.dtype = np.float32
    conf_tresh.shape = [1]

    nms = graph.non_max_suppression(
        boxes,
        graph.transpose(
            confidences, axis=np.asarray((0, 2, 1), dtype=np.int32)
        ),  # transpose confidences [1, num_det, 1] to [1, 1, num_det]
        max_output_boxes_per_class=topk,
        iou_threshold=iou_tresh,
        score_threshold=conf_tresh,
    )  # perform NMS using boxes and confidences as input
    nms.name = "NMS"
    nms.dtype = np.int64

    nms_out = graph.cast(
        graph.transpose(
            graph.gather(
                nms, indices=np.asarray([2], dtype=np.int32), axis=1
            ),  # gether selected boxes index from NMS [n, 1]
            axis=np.asarray((1, 0), dtype=np.int32),
        ),  # [1, n]
        to=6,
    )  # cast to int 32
    nms_out.name = "selected_idx"
    nms_out.dtype = np.int32
    nms_out.shape = [1, None]

    graph.inputs = [input_, config]
    graph.outputs = [nms_out]

    graph.cleanup().toposort()
    graph.fold_constants().cleanup()
    model = gs.export_onnx(graph)  # GS export to onnx
    onnx.checker.check_model(model)  # check onnx model

    print("Simplifying model...")
    model, check = simplify(gs.export_onnx(graph))
    assert check, "Simplified ONNX model could not be validated"

    yolov5 = ort.InferenceSession("yolov5n.onnx")
    nms = ort.InferenceSession(model.SerializeToString())
    nms_config = np.asarray([100, 0.45, 0.2], dtype=np.float32)

    img = cv2.imread("zidane.jpg")
    img = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=False, crop=False)

    output = yolov5.run(None, {"images": img})
    test_output = nms.run(None, {"detection": output[0], "config": nms_config})

    print(output[0][:, test_output[0][0], :].shape)

    onnx.save(model, "nms-yolov5.onnx")  # saving onnx model
