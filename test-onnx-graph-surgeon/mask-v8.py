import cv2
import numpy as np

import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort

from onnxsim import simplify


def get_divable_size(imgsz, stride):
    for i in range(len(imgsz)):
        div, mod = divmod(imgsz[i], stride)
        if mod > stride / 2:
            div += 1
        imgsz[i] = div * stride
    return imgsz


@gs.Graph.register()
def slice_(self, data, start, end, axis, steps=None):
    if steps:
        return self.layer(
            op="Slice", inputs=[data, start, end, axis, steps], outputs=["slice_out_gs"]
        )[0]
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
def add(self, a, b):
    return self.layer(
        op="Add",
        inputs=[a, b],
        outputs=["add_out_gs"],
    )[0]


@gs.Graph.register()
def sub(self, a, b):
    return self.layer(
        op="Sub",
        inputs=[a, b],
        outputs=["sub_out_gs"],
    )[0]


@gs.Graph.register()
def mul(self, a, b):
    return self.layer(
        op="Mul",
        inputs=[a, b],
        outputs=["mul_out_gs"],
    )[0]


@gs.Graph.register()
def div(self, a, b):
    return self.layer(
        op="Div",
        inputs=[a, b],
        outputs=["div_out_gs"],
    )[0]


@gs.Graph.register()
def matmul(self, a, b):
    return self.layer(
        op="MatMul",
        inputs=[a, b],
        outputs=["matmul_out_gs"],
    )[0]


@gs.Graph.register()
def round_(self, data):
    return self.layer(
        op="Round",
        inputs=[data],
        outputs=["round_out_gs"],
    )[0]


@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(
        op="Reshape",
        inputs=[data, shape],
        outputs=["reshape_out_gs"],
    )[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(
        op="Concat",
        inputs=inputs,
        outputs=["concat_out_gs"],
        attrs={"axis": axis},
    )[0]


@gs.Graph.register()
def sigmoid(self, data):
    return self.layer(
        op="Sigmoid",
        inputs=[data],
        outputs=["sigmoid_out_gs"],
    )[0]


@gs.Graph.register()
def resize(self, data, size, mode="linear"):
    return self.layer(
        op="Resize",
        inputs=[data, gs.Variable.empty(), gs.Variable.empty(), size],
        attrs={"mode": mode},
        outputs=["resize_out_gs"],
    )[0]


@gs.Graph.register()
def greater_equal(self, a, b):
    return self.layer(
        op="GreaterOrEqual",
        inputs=[a, b],
        outputs=["greaterequal_out_gs"],
    )[0]


@gs.Graph.register()
def pad(self, data, pads):
    return self.layer(
        op="Pad",
        inputs=[data, pads],
        outputs=["pad_out_gs"],
    )[0]


@gs.Graph.register()
def where(self, cond, x, y):
    return self.layer(
        op="Where",
        inputs=[cond, x, y],
        outputs=["where_out_gs"],
    )[0]


if __name__ == "__main__":
    graph = gs.Graph(opset=17)

    input_ = gs.Variable(name="detection", dtype=np.float32, shape=(None,))
    mask_det = gs.Variable(name="mask", dtype=np.float32, shape=(1, None, None, None))
    mask_config = gs.Variable(name="config", dtype=np.float32, shape=(9,))
    overlay = gs.Variable(name="overlay", dtype=np.uint8, shape=(None, None, 4))

    mask_det_shape = graph.cast(graph.shape(mask_det), to=1)
    mask_chanel = graph.slice_(
        mask_det_shape,
        start=np.asarray([1], dtype=np.int32),
        end=np.asarray([2], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    mask_height = graph.slice_(
        mask_det_shape,
        start=np.asarray([2], dtype=np.int32),
        end=np.asarray([3], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    mask_width = graph.slice_(
        mask_det_shape,
        start=np.asarray([3], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )

    x = graph.slice_(
        input_,
        start=np.asarray([0], dtype=np.int32),
        end=np.asarray([1], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    y = graph.slice_(
        input_,
        start=np.asarray([1], dtype=np.int32),
        end=np.asarray([2], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    w = graph.slice_(
        input_,
        start=np.asarray([2], dtype=np.int32),
        end=np.asarray([3], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    h = graph.slice_(
        input_,
        start=np.asarray([3], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    det_mask = graph.reshape(
        graph.slice_(
            input_,
            start=np.asarray([4], dtype=np.int32),
            end=graph.cast(graph.shape(input_), to=6),
            axis=np.asarray([0], dtype=np.int32),
        ),
        [1, -1],
    )

    max_size = graph.slice_(
        mask_config,
        start=np.asarray([0], dtype=np.int32),
        end=np.asarray([1], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
    )
    upsample_size = graph.cast(
        graph.slice_(
            mask_config,
            start=np.asarray([1], dtype=np.int32),
            end=np.asarray([5], dtype=np.int32),
            axis=np.asarray([0], dtype=np.int32),
        ),
        to=7,
    )
    target_size = graph.slice_(
        upsample_size,
        start=np.asarray([-1], dtype=np.int32),
        end=np.asarray([-3], dtype=np.int32),
        axis=np.asarray([0], dtype=np.int32),
        steps=np.asarray([-1], dtype=np.int32),
    )

    color = graph.cast(
        graph.slice_(
            mask_config,
            start=np.asarray([5], dtype=np.int32),
            end=np.asarray([9], dtype=np.int32),
            axis=np.asarray([0], dtype=np.int32),
        ),
        to=2,
    )

    x_reduced = graph.cast(graph.round_(graph.div(graph.mul(x, mask_width), max_size)), to=6)
    y_reduced = graph.cast(graph.round_(graph.div(graph.mul(y, mask_height), max_size)), to=6)
    w_reduced = graph.cast(graph.round_(graph.div(graph.mul(w, mask_width), max_size)), to=6)
    h_reduced = graph.cast(graph.round_(graph.div(graph.mul(h, mask_height), max_size)), to=6)

    crop_protos = graph.reshape(
        graph.slice_(
            graph.slice_(
                mask_det,
                start=y_reduced,
                end=graph.add(y_reduced, h_reduced),
                axis=np.asarray([2], dtype=np.int32),
            ),
            start=x_reduced,
            end=graph.add(x_reduced, w_reduced),
            axis=np.asarray([3], dtype=np.int32),
        ),
        graph.concat([graph.cast(mask_chanel, to=7), [-1]]),
    )

    protos = graph.reshape(
        graph.resize(
            graph.reshape(
                graph.sigmoid(graph.matmul(det_mask, crop_protos)),
                graph.cast(graph.concat([h_reduced, w_reduced]), to=7),
            ),
            target_size,
        ),
        graph.concat([target_size, [1]]),
    )

    max_size_int64 = graph.cast(max_size, to=7)
    pad_size = graph.concat(
        [
            graph.slice_(
                upsample_size,
                start=np.asarray([1], dtype=np.int32),
                end=np.asarray([2], dtype=np.int32),
                axis=np.asarray([0], dtype=np.int32),
            ),
            graph.slice_(
                upsample_size,
                start=np.asarray([0], dtype=np.int32),
                end=np.asarray([1], dtype=np.int32),
                axis=np.asarray([0], dtype=np.int32),
            ),
            np.asarray([0], dtype=np.int64),
            graph.sub(
                max_size_int64,
                graph.add(
                    graph.slice_(
                        upsample_size,
                        start=np.asarray([1], dtype=np.int32),
                        end=np.asarray([2], dtype=np.int32),
                        axis=np.asarray([0], dtype=np.int32),
                    ),
                    graph.slice_(
                        upsample_size,
                        start=np.asarray([3], dtype=np.int32),
                        end=np.asarray([4], dtype=np.int32),
                        axis=np.asarray([0], dtype=np.int32),
                    ),
                ),
            ),
            graph.sub(
                max_size_int64,
                graph.add(
                    graph.slice_(
                        upsample_size,
                        start=np.asarray([0], dtype=np.int32),
                        end=np.asarray([1], dtype=np.int32),
                        axis=np.asarray([0], dtype=np.int32),
                    ),
                    graph.slice_(
                        upsample_size,
                        start=np.asarray([2], dtype=np.int32),
                        end=np.asarray([3], dtype=np.int32),
                        axis=np.asarray([0], dtype=np.int32),
                    ),
                ),
            ),
            np.asarray([0], dtype=np.int64),
        ]
    )

    mask_filter = graph.where(
        graph.pad(graph.greater_equal(protos, np.asarray([0.5], dtype=np.float32)), pad_size),
        color,
        overlay,
    )
    mask_filter.dtype = np.uint8
    mask_filter.shape = [None, None, 4]
    mask_filter.name = "mask_filter"

    graph.inputs = [input_, mask_det, mask_config, overlay]
    graph.outputs = [mask_filter]

    graph.cleanup().toposort()
    graph.fold_constants().cleanup()
    model = gs.export_onnx(graph)  # GS export to onnx
    onnx.checker.check_model(model)  # check onnx model

    model, check = simplify(gs.export_onnx(graph))
    assert check, "Simplified ONNX model could not be validated"

    yolov8_seg = ort.InferenceSession("./yolov8n-seg.onnx")
    nms = ort.InferenceSession("./nms-yolov8.onnx")
    mask_sess = ort.InferenceSession(model.SerializeToString())

    n_class = 80
    topk = 100
    iou_threshold = 0.45
    score_threshold = 0.2

    img = cv2.imread("zidane.jpg")
    nms_config = np.asarray([n_class, topk, iou_threshold, score_threshold], dtype=np.float32)

    source_height, source_width, _ = img.shape
    source_width, source_height = get_divable_size([source_width, source_height], 32)
    source = cv2.resize(img, [source_width, source_height])

    ## padding image
    max_size = max(source_width, source_height)  # get max size
    source_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros mat
    source_padded[:source_height, :source_width] = source.copy()  # place original image

    ## ratios
    x_ratio = max_size / 640
    y_ratio = max_size / 640

    input_ = cv2.dnn.blobFromImage(source_padded, 1 / 255.0, (640, 640), swapRB=False, crop=False)

    output = yolov8_seg.run(None, {"images": input_})
    selected = nms.run(None, {"detection": output[0], "config": nms_config})[0]

    all_mask = np.zeros([max_size, max_size, 4], dtype=np.uint8)

    for i in range(selected.shape[1]):
        box = selected[0, i, :4]
        score = selected[0, i, 4 : 4 + n_class].max()
        label = np.argmax(selected[0, i, 4 : 4 + n_class])
        masked = selected[0, i, 4 + n_class :]

        box[0] = (box[0] - 0.5 * box[2]) * x_ratio
        box[1] = (box[1] - 0.5 * box[3]) * y_ratio
        box[2] *= x_ratio
        box[3] *= y_ratio

        box = box.round().astype(np.int32)

        # crop mask from proto
        x = int(round(box[0] * 160 / max_size))
        y = int(round(box[1] * 160 / max_size))
        w = int(round(box[2] * 160 / max_size))
        h = int(round(box[3] * 160 / max_size))

        # process protos
        protos = output[1][0, :, y : y + h, x : x + w].reshape(32, -1)
        protos = np.matmul(np.expand_dims(masked, 0), protos)  # matmul
        protos = 1 / (1 + np.exp(-protos))  # sigmoid
        protos = protos.reshape(h, w)  # reshape
        mask = cv2.resize(protos, (box[2], box[3]))  # resize mask
        mask = np.expand_dims(mask, -1)
        mask = np.where(mask > 0.5, [255, 255, 255, 255], [0, 0, 0, 0])
        mask = np.pad(
            mask,
            [[box[1], max_size - box[1] - box[3]], [box[0], max_size - box[0] - box[2]], [0, 0]],
        )

        test_out = mask_sess.run(
            None,
            {
                "detection": np.asarray([*box, *masked], dtype=np.float32),
                "mask": output[1],
                "config": np.asarray([max_size, *box, 255, 255, 255, 255], dtype=np.float32),
                "overlay": all_mask,
            },
        )

        # print(np.equal(test_out[0], protos).all())
        print(test_out[0].shape)
        print(mask.shape)

        # cv2.imwrite(f"output/pred{i}.png", test_out[0])
        # cv2.imwrite(f"output/val{i}.png", mask)

        all_mask = test_out[0]

    cv2.imwrite(f"val.png", all_mask)
    onnx.save(model, "mask-yolov8-seg.onnx")  # saving onnx model
