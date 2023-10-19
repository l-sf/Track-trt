echo ""
echo ""
echo "************************ compile LightTrack models ***************************"
echo ""
trtexec --onnx=./onnx_models/lighttrack-z.onnx \
		    --saveEngine=./lighttrack-z.trt \
		    --buildOnly \
		    --fp16

echo ""
trtexec --onnx=./onnx_models/lighttrack-x-head.onnx \
		    --saveEngine=./lighttrack-x-head.trt \
		    --buildOnly \
		    --fp16

echo ""
echo ""
echo "************************ compile StarkLightning models ***************************"
echo ""
trtexec --onnx=./onnx_models/starklight-z.onnx \
		    --saveEngine=./starklight-z.trt \
		    --buildOnly \
		    --fp16

echo ""
trtexec --onnx=./onnx_models/starklight-x-head.onnx \
		    --saveEngine=./starklight-x-head.trt \
		    --buildOnly \
		    --fp16


echo ""
echo ""
echo "************************ compile OSTrack models ***************************"
echo ""
trtexec --onnx=./onnx_models/ostrack-256.onnx \
		    --saveEngine=./ostrack-256.trt \
		    --buildOnly \
		    --fp16

echo ""
trtexec --onnx=./onnx_models/ostrack-384-ce.onnx \
		    --saveEngine=./ostrack-384-ce.trt \
		    --buildOnly \
		    --fp16

