import { Interpreter, Tensor } from "node-tflite";
import fs from "fs";
import path from "path";

function canvasToRGBFloat(context: CanvasRenderingContext2D) {
    const { width, height } = context.canvas;
    const data = context.getImageData(0, 0, width, height);

    const rgbFloat = new Float32Array(width * height * 3);

    for (let i = 0; i < width * height; ++i) {
        rgbFloat[i * 3] = data.data[i * 4] / 255;
        rgbFloat[i * 3 + 1] = data.data[i * 4 + 1] / 255;
        rgbFloat[i * 3 + 2] = data.data[i * 4 + 2] / 255;
    }

    return rgbFloat;
}

function sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x));
}


class TFLiteModel {
    protected inputSize = 128;
    protected inputCanvas: HTMLCanvasElement;
    protected inputContext: CanvasRenderingContext2D;
    protected interpreter: Interpreter;

    constructor(model_path:string) {

        this.inputCanvas = document.createElement("canvas");
        this.inputCanvas.width = this.inputSize;
        this.inputCanvas.height = this.inputSize;
        this.inputContext = this.inputCanvas.getContext("2d")!;

        const modelPath = path.resolve(
            __dirname,
            model_path
        );
        this.interpreter = new Interpreter(fs.readFileSync(modelPath));
        this.interpreter.allocateTensors();

        this.interpreter.allocateTensors();
        console.log(model_path + ": "+this.interpreter.inputs[0].dims)

    }

    forward(input: CanvasImageSource, inputSize:number|undefined = undefined) : Tensor[]{
        inputSize = (inputSize || this.inputSize);
        if (inputSize != this.inputSize){
            this.inputSize = inputSize;
            this.inputCanvas = document.createElement("canvas");
            this.inputCanvas.width = this.inputSize;
            this.inputCanvas.height = this.inputSize;
            this.inputContext = this.inputCanvas.getContext("2d")!;
        }

        // console.log("before draw image" + Date.now() + Date());
        this.inputContext.drawImage(input, 0, 0, inputSize, inputSize);
        const rgbFloat = canvasToRGBFloat(this.inputContext);

        // console.log("after draw image" + Date.now() + Date());


        // console.log(inputSize, interpreter.inputs[0].dims, rgbFloat);
        this.interpreter.inputs[0].copyFrom(rgbFloat);

        console.log("before interpreter invoke" + Date.now() + Date());
        this.interpreter.invoke();
        console.log("after interpreter invoke" + Date.now() + Date());

        return this.interpreter.outputs;
    }
}

type Box = [number, number, number, number]; // left, top, right, bottom
class FaceDetector extends TFLiteModel{
    private generateAnchors(width: number, height: number): [number, number][] {
        const outputSpec = {
            strides: [8, 16] as const,
            anchors: [2, 6] as const,
        };

        const anchors: [number, number][] = [];
        for (let i = 0; i < outputSpec.strides.length; i++) {
            const stride = outputSpec.strides[i];
            const gridRows = Math.floor((height + stride - 1) / stride);
            const gridCols = Math.floor((width + stride - 1) / stride);
            const anchorsNum = outputSpec.anchors[i];

            for (let gridY = 0; gridY < gridRows; gridY++) {
                const anchorY = stride * (gridY + 0.5);

                for (let gridX = 0; gridX < gridCols; gridX++) {
                    const anchorX = stride * (gridX + 0.5);
                    for (let n = 0; n < anchorsNum; n++) {
                        anchors.push([anchorX, anchorY]);
                    }
                }
            }
        }

        return anchors;
    }

    detectFace(input: CanvasImageSource): Box | undefined {
        var anchors = this.generateAnchors(this.inputSize, this.inputSize)

        const coordinatesData = new Float32Array(anchors.length * 16);
        const scoreData = new Float32Array(anchors.length);

        var outputs = this.forward(input);
        outputs[0].copyTo(coordinatesData);
        outputs[1].copyTo(scoreData);

        for (let i = 0; i < anchors.length; ++i) {
            scoreData[i] = sigmoid(scoreData[i]);
        }

        const maxScore = Math.max(...Array.from(scoreData));
        if (maxScore < 0.75) {
            return;
        }

        // Find up to 1 faces
        const bestIndex = scoreData.indexOf(maxScore);

        const centerX =
            coordinatesData[bestIndex * 16] + anchors[bestIndex][0];
        const centerY =
            coordinatesData[bestIndex * 16 + 1] + anchors[bestIndex][1];
        const width = coordinatesData[bestIndex * 16 + 2];
        const height = coordinatesData[bestIndex * 16 + 3];
        const left = centerX - width / 2;
        const top = centerY - height / 2;
        const right = left + width;
        const bottom = top + height;

        return [
            (left / this.inputSize) * (input.width as number),
            (top / this.inputSize) * (input.height as number),
            (right / this.inputSize) * (input.width as number),
            (bottom / this.inputSize) * (input.height as number),
        ];
    }

}

class PortraitMasker extends TFLiteModel{
    maskPortrait(input: CanvasImageSource): Float32Array{
        let outputs = this.forward(input, this.interpreter.inputs[0].dims[1]);
        let outputSize = outputs[0].dims.reduce((previousValue, currentValue) => previousValue * currentValue);
        var mask = new Float32Array(outputSize);
        // console.log(outputSize, outputs[0].dims);
        outputs[0].copyTo(mask);
        return mask;
    }
}


async function init() {
    const video = document.createElement("video");
    video.width = 256;
    video.height = 256;
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            width: { ideal: video.width },
            height: { ideal: video.height },
        },
    });
    video.srcObject = stream;
    video.play();

    const faceDetector = new FaceDetector("face_detection_front.tflite");
    const masker = new PortraitMasker("mlkit.tflite");

    const canvas = document.getElementById("canvas") as HTMLCanvasElement;
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext("2d")!;
    context.strokeStyle = "red";
    context.lineWidth = 2;
    
    var pre_mask: Float32Array;
    var mask : Float32Array;
    var pre_video : HTMLVideoElement;

    const animate = () => {

        // console.log("Before inference:" + Date.now() + Date());

        var next_mask = masker.maskPortrait(video);

        // console.log("After inference:" + Date.now() + Date());

        if (pre_mask && mask && pre_video) {
            context.drawImage(pre_video, 0, 0);

            const colorMap = context.getImageData(0, 0, canvas.width, canvas.height).data;

            for (var i = 0; i < colorMap.length; i += 4) {
                var blinkThrehold = 0.1;
                let isHumanThrehold = 0.5;
                var sideSimilar = Math.abs(pre_mask[i / 4] - next_mask[i / 4]) <= blinkThrehold;
                var midUnsimilar = Math.abs(mask[i / 4] - pre_mask[i / 4]) > blinkThrehold && Math.abs(mask[i / 4] - next_mask[i / 4]) > blinkThrehold;
                var isHuman = 1;

                if (sideSimilar && midUnsimilar) {
                    isHuman = (pre_mask[i / 4] + next_mask[i / 4]) / 2;
                }else {
                    isHuman = mask[i / 4];
                }

                if (isHuman < isHumanThrehold) {
                    colorMap[i + 3] = 0  // A value
                }else{
                    colorMap[i + 3] = 255; // A value;
                }
            }
            context.putImageData(new ImageData(colorMap, canvas.width, canvas.height), 0, 0);
        }
        pre_video = video;
        pre_mask = mask;
        mask = next_mask;

        const rect = faceDetector.detectFace(video);
        if (rect) {
            context.strokeRect(
                rect[0],
                rect[1],
                rect[2] - rect[0],
                rect[3] - rect[1]
            );
        }

        requestAnimationFrame(animate);
    };
    animate();
}

init();




