"�.
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE133333�@A33333�@a8R��?i8R��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff:�@9fffff:�@Afffff:�@Ifffff:�@aN�Wپ�?i(���:�?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     �i@9     �i@A     �i@I     �i@a�U����?i��)���?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(133333Q@933333Q@A33333Q@I33333Q@a���Ɠ?i�r�.��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����YK@9�����YK@A�����YK@I�����YK@aR ����?i���$�1�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����YF@9�����YF@A�����YF@I�����YF@a��;p��?iʕ�p��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1������@@9������@@A������@@I������@@a1�1�:�?i�[�PX��?�Unknown
o	Host_FusedMatMul"sequential/dense/Relu(133333�9@933333�9@A33333�9@I33333�9@a�;)�J�}?in����!�?�Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1�����L5@9�����L5@A�����L5@I�����L5@a�/���x?i��48S�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffff.@9ffffff.@Affffff.@Iffffff.@ap��q?i�9nv�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff)@9ffffff)@Affffff)@Iffffff)@a��EWkm?iQI<nٓ�?�Unknown
iHostWriteSummary"WriteSummary(1333333&@9333333&@A333333&@I333333&@a���y��i?iI�����?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff%@9ffffff%@Affffff%@Iffffff%@a�kWM�h?iV~>Y��?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1333333"@9333333"@A333333"@I333333"@aa��xe?i�m۶m��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������!@9������!@A������!@I������!@a�0(��bd?i��G���?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1333333@9333333@A333333@I333333@a��:�b?i������?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff"@9ffffff"@Affffff@Iffffff@ap��a?i>�z�|�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������@9������@A������@I������@a����V$a?i9�v�$�?�Unknown
dHostDataset"Iterator::Model(1333333#@9333333#@A333333@I333333@a�����Z?i��2�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������@9������@A333333@I333333@a�����Z?i�x[!�?�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1333333@9333333@A333333@I333333@a�����Z?i&�ͣ�L�?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a2gt�ʠW?iZD	�X�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a��8�<V?i]����c�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1333333@9333333@A333333@I333333@a��8�<V?i`�|�n�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@aM���`�U?i���4�y�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@a�0(��bT?i��J}��?�Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a�Zj�&�R?i�U��?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a�Zj�&�R?iS�����?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@ap��Q?i���g��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a����V$Q?i��>H���?�Unknown�
s HostCast"!sequential/dropout_1/dropout/Cast(1������	@9������	@A������	@I������	@a^^�K��M?i(�Q�b��?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@aks~)o�L?i�M���?�Unknown
�"HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aks~)o�L?ib}樿��?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ax�:�K?iDEh����?�Unknown
`$HostGatherV2"
GatherV2_1(1333333@9333333@A333333@I333333@a�����J?iku�xj��?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a�����J?i���9"��?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�ܣ}eH?i�:(��?�Unknown
l'HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a��D[0*G?i��P����?�Unknown
x(HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������4@9������4@A333333@I333333@a��8�<F?iGށ��?�Unknown
o)HostMul"sequential/dropout/dropout/Mul(1333333@9333333@A333333@I333333@a��8�<F?i�R���?�Unknown
�*HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1333333@9333333@A333333@I333333@a��8�<F?iK��[���?�Unknown
�+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff�?9ffffff�?Affffff�?Iffffff�?ap��A?i'���?�Unknown
�,HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a��j��@?iHz9�2��?�Unknown
�-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1�������?9�������?A�������?I�������?a^^�K��=?i���[���?�Unknown
�.HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      "@9      "@A�������?I�������?a�ܣ}e8?ip�rH���?�Unknown
a/HostIdentity"Identity(1�������?9�������?A�������?I�������?a��j��0?i      �?�Unknown�*�-
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff:�@9fffff:�@Afffff:�@Ifffff:�@a���Қ�?i���Қ�?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     �i@9     �i@A     �i@I     �i@a@Me-�w�?i��:8�F�?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(133333Q@933333Q@A33333Q@I33333Q@a�9]�x��?iDw��V�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����YK@9�����YK@A�����YK@I�����YK@aD���.�?i�o��/�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����YF@9�����YF@A�����YF@I�����YF@ac;�.�6�?i�!L�I��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1������@@9������@@A������@@I������@@a��h��?iό�Fe�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(133333�9@933333�9@A33333�9@I33333�9@a	~R���?iR���r��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1�����L5@9�����L5@A�����L5@I�����L5@alj{'u+�?i��t�  �?�Unknown
�	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffff.@9ffffff.@Affffff.@Iffffff.@a����6~?i�|�9�\�?�Unknown
}
HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff)@9ffffff)@Affffff)@Iffffff)@a����>y?i�����?�Unknown
iHostWriteSummary"WriteSummary(1333333&@9333333&@A333333&@I333333&@aү��rv?i���f,��?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff%@9ffffff%@Affffff%@Iffffff%@a�G��Du?i4�:4���?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1333333"@9333333"@A333333"@I333333"@a���Եr?i�+��	�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������!@9������!@A������!@I������!@at�
�~q?iA<��,�?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1333333@9333333@A333333@I333333@a�eAH\o?i����K�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff"@9ffffff"@Affffff@Iffffff@a����6n?iSf��j�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������@9������@A������@I������@a�?��Ckm?i��q)���?�Unknown
dHostDataset"Iterator::Model(1333333#@9333333#@A333333@I333333@a���(�g?i;�����?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������@9������@A333333@I333333@a���(�g?i������?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1333333@9333333@A333333@I333333@a���(�g?i���ϰ��?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a%RjwFd?i�VG���?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a~��%c?ix�nl��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1333333@9333333@A333333@I333333@a~��%c?i@���!�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a����^�b?i?.����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1������@9������@A������@I������@at�
�~a?i�81�N+�?�Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a���~��_?i]�p�5;�?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a���~��_?i�ׯ�K�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@a����6^?i�ɸM8Z�?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a�?��Ck]?i⌦��h�?�Unknown�
sHostCast"!sequential/dropout_1/dropout/Cast(1������	@9������	@A������	@I������	@a�`�ˆqY?if��u�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@a��T���X?iyW����?�Unknown
� HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��T���X?iມ�L��?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�:�^n�W?i}6��9��?�Unknown
`"HostGatherV2"
GatherV2_1(1333333@9333333@A333333@I333333@a���(�W?iQ��U���?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a���(�W?i%���H��?�Unknown
�$HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�=�T?i���垻�?�Unknown
l%HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a�[#O��S?iJ"d>���?�Unknown
x&HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������4@9������4@A333333@I333333@a~��%S?i.�����?�Unknown
o'HostMul"sequential/dropout/dropout/Mul(1333333@9333333@A333333@I333333@a~��%S?i�|c���?�Unknown
�(HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1333333@9333333@A333333@I333333@a~��%S?i�J	�.��?�Unknown
�)HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff�?9ffffff�?Affffff�?Iffffff�?a����6N?i������?�Unknown
�*HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1�������?9�������?A�������?I�������?a��(���L?i�����?�Unknown
�+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1�������?9�������?A�������?I�������?a�`�ˆqI?i����@��?�Unknown
�,HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      "@9      "@A�������?I�������?a�=�D?i�Z	l��?�Unknown
a-HostIdentity"Identity(1�������?9�������?A�������?I�������?a��(���<?i      �?�Unknown�