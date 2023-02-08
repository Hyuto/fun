import cv2
import numpy as np

import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort

from onnxsim import simplify


@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])[0]


@gs.Graph.register()
def slice(self, data, start, end, axis=2):
    return self.layer(op="Slice", inputs=[data, start, end, axis], outputs=["slice_out_gs"])[0]


@gs.Graph.register()
def shape(self, data):
    return self.layer(op="Shape", inputs=[data], outputs=["shape_out_gs"])[0]


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
def reduce_max(self, data, axis, keepdims):
    return self.layer(
        op="ReduceMax",
        inputs=[data],
        outputs=["reduce-max_out_gs"],
        attrs={"axes": axis, "keepdims": keepdims},
    )[0]


@gs.Graph.register()
def squeeze(self, data, axis):
    return self.layer(
        op="Squeeze",
        inputs=[data],
        attrs={"axes": axis},
        outputs=["squeeze_out_gs"],
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
    config = gs.Variable(name="config", dtype=np.float32, shape=(4,))

    num_class = graph.cast(
        graph.slice(
            config,
            start=np.asarray([0], dtype=np.int32),
            end=np.asarray([1], dtype=np.int32),
            axis=np.asarray([0], dtype=np.int32),
        ),
        to=6,
    )
    num_class.name = "num-class"
    num_class.dtype = np.int32
    num_class.shape = [1]

    topk = graph.cast(
        graph.slice(
            config,
            start=np.asarray([1], dtype=np.int32),
            end=np.asarray([2], dtype=np.int32),
            axis=np.asarray([0], dtype=np.int32),
        ),  # slice topk from outputs
        to=7,
    )  # cast to int64
    topk.name = "topk"
    topk.dtype = np.int64
    topk.shape = [1]

    iou_tresh = graph.slice(
        config,
        start=np.asarray([2], dtype=np.int32),
        end=np.asarray([3], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )  # slice iou_tresh from outputs
    iou_tresh.name = "iou_tresh"
    iou_tresh.dtype = np.float32
    iou_tresh.shape = [1]

    score_tresh = graph.slice(
        config,
        start=np.asarray([3], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )  # slice score_tresh from outputs
    score_tresh.name = "score_tresh"
    score_tresh.dtype = np.float32
    score_tresh.shape = [1]

    input_T = graph.transpose(input_, axis=np.asarray((0, 2, 1), dtype=np.int32))

    boxes = graph.slice(
        input_T,
        start=np.asarray([0], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice boxes from outputs
    boxes.name = "raw-boxes"
    boxes.dtype = np.float32
    boxes.shape = [1, None, 4]

    scores = graph.slice(
        input_T,
        start=np.asarray([4], dtype=np.int32),
        end=graph.add(num_class, np.asarray([4], dtype=np.int32)),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice confidences from outputs
    scores.name = "raw-scores"
    scores.dtype = np.float32
    scores.shape = [1, None, None]

    max_scores = graph.reduce_max(
        scores, axis=np.asarray([2], dtype=np.int32), keepdims=1
    )  # slice confidences from outputs
    max_scores.name = "max-scores"
    max_scores.dtype = np.float32
    max_scores.shape = [1, None, 1]

    nms = graph.non_max_suppression(
        boxes,
        graph.transpose(
            max_scores, axis=np.asarray((0, 2, 1), dtype=np.int32)
        ),  # transpose confidences [1, num_det, 1] to [1, 1, num_det]
        max_output_boxes_per_class=topk,
        iou_threshold=iou_tresh,
        score_threshold=score_tresh,
    )  # perform NMS using boxes and confidences as input
    nms.name = "NMS"
    nms.dtype = np.int64

    idx = graph.transpose(
        graph.gather(
            nms, indices=np.asarray([2], dtype=np.int32), axis=1
        ),  # gether selected boxes index from NMS
        axis=np.asarray((1, 0), dtype=np.int32),
    )  # transpose index from [n, 1] to [1, n]
    idx.dtype = np.int64

    selected = graph.squeeze(
        graph.gather(input_T, indices=idx, axis=1),  # indexing boxes
        axis=[1],
    )  # squeeze tensor dimension [1, 1, n, 4] to [1, n, 4]
    selected.name = "selected"
    selected.dtype = np.float32
    selected.shape = [1, None, None]

    graph.inputs = [input_, config]
    graph.outputs = [selected, scores]

    graph.cleanup().toposort()
    graph.fold_constants().cleanup()
    model = gs.export_onnx(graph)  # GS export to onnx
    onnx.checker.check_model(model)  # check onnx model

    print("Simplifying model...")
    model, check = simplify(gs.export_onnx(graph))
    assert check, "Simplified ONNX model could not be validated"

    yolov5 = ort.InferenceSession("yolov8n.onnx")
    nms = ort.InferenceSession(model.SerializeToString())
    nms_config = np.asarray([80, 100, 0.45, 0.2], dtype=np.float32)

    img = cv2.imread("zidane.jpg")
    source_height, source_width, _ = img.shape

    ## padding image
    max_size = max(source_width, source_height)  # get max size
    source_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros mat
    source_padded[:source_height, :source_width] = img.copy()  # place original image

    source_padded = cv2.dnn.blobFromImage(
        source_padded, 1 / 255.0, (640, 640), swapRB=False, crop=False
    )

    output = yolov5.run(None, {"images": source_padded})
    test_output = nms.run(None, {"detection": output[0], "config": nms_config})

    print(test_output[0].shape)

    onnx.save(model, "nms-yolov8.onnx")  # saving onnx model
