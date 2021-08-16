"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var node_tflite_1 = require("node-tflite");
var fs_1 = __importDefault(require("fs"));
var path_1 = __importDefault(require("path"));
function canvasToRGBFloat(context) {
    var _a = context.canvas, width = _a.width, height = _a.height;
    var data = context.getImageData(0, 0, width, height);
    var rgbFloat = new Float32Array(width * height * 3);
    for (var i = 0; i < width * height; ++i) {
        rgbFloat[i * 3] = data.data[i * 4] / 255;
        rgbFloat[i * 3 + 1] = data.data[i * 4 + 1] / 255;
        rgbFloat[i * 3 + 2] = data.data[i * 4 + 2] / 255;
    }
    return rgbFloat;
}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
var TFLiteModel = /** @class */ (function () {
    function TFLiteModel(model_path) {
        this.inputSize = 128;
        this.inputCanvas = document.createElement("canvas");
        this.inputCanvas.width = this.inputSize;
        this.inputCanvas.height = this.inputSize;
        this.inputContext = this.inputCanvas.getContext("2d");
        var modelPath = path_1.default.resolve(__dirname, model_path);
        this.interpreter = new node_tflite_1.Interpreter(fs_1.default.readFileSync(modelPath));
        this.interpreter.allocateTensors();
        this.interpreter.allocateTensors();
        console.log(model_path + ": " + this.interpreter.inputs[0].dims);
    }
    TFLiteModel.prototype.forward = function (input, inputSize) {
        if (inputSize === void 0) { inputSize = undefined; }
        inputSize = (inputSize || this.inputSize);
        if (inputSize != this.inputSize) {
            this.inputSize = inputSize;
            this.inputCanvas = document.createElement("canvas");
            this.inputCanvas.width = this.inputSize;
            this.inputCanvas.height = this.inputSize;
            this.inputContext = this.inputCanvas.getContext("2d");
        }
        // console.log("before draw image" + Date.now() + Date());
        this.inputContext.drawImage(input, 0, 0, inputSize, inputSize);
        var rgbFloat = canvasToRGBFloat(this.inputContext);
        // console.log("after draw image" + Date.now() + Date());
        // console.log(inputSize, interpreter.inputs[0].dims, rgbFloat);
        this.interpreter.inputs[0].copyFrom(rgbFloat);
        console.log("before interpreter invoke" + Date.now() + Date());
        this.interpreter.invoke();
        console.log("after interpreter invoke" + Date.now() + Date());
        return this.interpreter.outputs;
    };
    return TFLiteModel;
}());
var FaceDetector = /** @class */ (function (_super) {
    __extends(FaceDetector, _super);
    function FaceDetector() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    FaceDetector.prototype.generateAnchors = function (width, height) {
        var outputSpec = {
            strides: [8, 16],
            anchors: [2, 6],
        };
        var anchors = [];
        for (var i = 0; i < outputSpec.strides.length; i++) {
            var stride = outputSpec.strides[i];
            var gridRows = Math.floor((height + stride - 1) / stride);
            var gridCols = Math.floor((width + stride - 1) / stride);
            var anchorsNum = outputSpec.anchors[i];
            for (var gridY = 0; gridY < gridRows; gridY++) {
                var anchorY = stride * (gridY + 0.5);
                for (var gridX = 0; gridX < gridCols; gridX++) {
                    var anchorX = stride * (gridX + 0.5);
                    for (var n = 0; n < anchorsNum; n++) {
                        anchors.push([anchorX, anchorY]);
                    }
                }
            }
        }
        return anchors;
    };
    FaceDetector.prototype.detectFace = function (input) {
        var anchors = this.generateAnchors(this.inputSize, this.inputSize);
        var coordinatesData = new Float32Array(anchors.length * 16);
        var scoreData = new Float32Array(anchors.length);
        var outputs = this.forward(input);
        outputs[0].copyTo(coordinatesData);
        outputs[1].copyTo(scoreData);
        for (var i = 0; i < anchors.length; ++i) {
            scoreData[i] = sigmoid(scoreData[i]);
        }
        var maxScore = Math.max.apply(Math, Array.from(scoreData));
        if (maxScore < 0.75) {
            return;
        }
        // Find up to 1 faces
        var bestIndex = scoreData.indexOf(maxScore);
        var centerX = coordinatesData[bestIndex * 16] + anchors[bestIndex][0];
        var centerY = coordinatesData[bestIndex * 16 + 1] + anchors[bestIndex][1];
        var width = coordinatesData[bestIndex * 16 + 2];
        var height = coordinatesData[bestIndex * 16 + 3];
        var left = centerX - width / 2;
        var top = centerY - height / 2;
        var right = left + width;
        var bottom = top + height;
        return [
            (left / this.inputSize) * input.width,
            (top / this.inputSize) * input.height,
            (right / this.inputSize) * input.width,
            (bottom / this.inputSize) * input.height,
        ];
    };
    return FaceDetector;
}(TFLiteModel));
var PortraitMasker = /** @class */ (function (_super) {
    __extends(PortraitMasker, _super);
    function PortraitMasker() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    PortraitMasker.prototype.maskPortrait = function (input) {
        var outputs = this.forward(input, this.interpreter.inputs[0].dims[1]);
        var outputSize = outputs[0].dims.reduce(function (previousValue, currentValue) { return previousValue * currentValue; });
        var mask = new Float32Array(outputSize);
        // console.log(outputSize, outputs[0].dims);
        outputs[0].copyTo(mask);
        return mask;
    };
    return PortraitMasker;
}(TFLiteModel));
function init() {
    return __awaiter(this, void 0, void 0, function () {
        var video, stream, faceDetector, masker, canvas, context, pre_mask, mask, pre_video, animate;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    video = document.createElement("video");
                    video.width = 256;
                    video.height = 256;
                    return [4 /*yield*/, navigator.mediaDevices.getUserMedia({
                            audio: false,
                            video: {
                                width: { ideal: video.width },
                                height: { ideal: video.height },
                            },
                        })];
                case 1:
                    stream = _a.sent();
                    video.srcObject = stream;
                    video.play();
                    faceDetector = new FaceDetector("face_detection_front.tflite");
                    masker = new PortraitMasker("mlkit.tflite");
                    canvas = document.getElementById("canvas");
                    canvas.width = 256;
                    canvas.height = 256;
                    context = canvas.getContext("2d");
                    context.strokeStyle = "red";
                    context.lineWidth = 2;
                    animate = function () {
                        // console.log("Before inference:" + Date.now() + Date());
                        var next_mask = masker.maskPortrait(video);
                        // console.log("After inference:" + Date.now() + Date());
                        if (pre_mask && mask && pre_video) {
                            context.drawImage(pre_video, 0, 0);
                            var colorMap = context.getImageData(0, 0, canvas.width, canvas.height).data;
                            for (var i = 0; i < colorMap.length; i += 4) {
                                var blinkThrehold = 0.1;
                                var isHumanThrehold = 0.5;
                                var sideSimilar = Math.abs(pre_mask[i / 4] - next_mask[i / 4]) <= blinkThrehold;
                                var midUnsimilar = Math.abs(mask[i / 4] - pre_mask[i / 4]) > blinkThrehold && Math.abs(mask[i / 4] - next_mask[i / 4]) > blinkThrehold;
                                var isHuman = 1;
                                if (sideSimilar && midUnsimilar) {
                                    isHuman = (pre_mask[i / 4] + next_mask[i / 4]) / 2;
                                }
                                else {
                                    isHuman = mask[i / 4];
                                }
                                if (isHuman < isHumanThrehold) {
                                    colorMap[i + 3] = 0; // A value
                                }
                                else {
                                    colorMap[i + 3] = 255; // A value;
                                }
                            }
                            context.putImageData(new ImageData(colorMap, canvas.width, canvas.height), 0, 0);
                        }
                        pre_video = video;
                        pre_mask = mask;
                        mask = next_mask;
                        var rect = faceDetector.detectFace(video);
                        if (rect) {
                            context.strokeRect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
                        }
                        requestAnimationFrame(animate);
                    };
                    animate();
                    return [2 /*return*/];
            }
        });
    });
}
init();
