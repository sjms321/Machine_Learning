"�I
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1�����	�@9�����	�@A�����	�@I�����	�@ae�b�6�?ie�b�6�?�Unknown�
BHostIDLE"IDLE1     ܅@A     ܅@a������?i]o����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����YY@9�����YY@A�����YY@I�����YY@a�d�d��?i���%���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�V@933333�V@A33333�V@I33333�V@a�H%�e�?i�6I����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      S@9      S@A      S@I      S@a�3(I�?i�
�)��?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(133333�J@933333�J@A33333�J@I33333�J@a�1�.���?i���ǀ�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333�H@933333�H@A33333�H@I33333�H@a/D˦y:�?i�R����?�Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(133333�D@933333�D@A33333�D@I33333�D@aO �M%�?i����ś�?�Unknown
d
HostDataset"Iterator::Model(1�����L3@9�����L3@A������,@I������,@a�82v?i��:*��?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      *@9      *@A      *@I      *@a49�+�	t?ig��X=��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      &@9      &@A      &@I      &@aS�E�u�p?iR��C&�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������#@9������#@A������#@I������#@ac��ӄn?iؚ-�0�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333&@9333333&@A      #@I      #@a�3(Im?i��E?�M�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a�bƑ�k?i@�Ѳi�?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1������!@9������!@A������!@I������!@a�B��ok?i���w"��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1333333 @9333333 @A333333 @I333333 @ah?�O�h?i������?�Unknown
iHostWriteSummary"WriteSummary(1       @9       @A       @I       @a�!e�h?i�-Ķ�?�Unknown�
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a�v*� 0b?iX�N���?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@a}V�6�a?i��0����?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@a*���`?iBc 9���?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      @9      @Affffff@Iffffff@a���$�q_?i��2���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a fޯ�5^?i鳊��	�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@aX�0Q�Z?iZL�j��?�Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a$۞}8�U?i��!�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      7@9      7@A      @I      @a$۞}8�U?i6�0��,�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a$۞}8�U?i��o�P7�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a^�,�b�T?i�P���A�?�Unknown
`HostGatherV2"
GatherV2_1(1������	@9������	@A������	@I������	@a�HN��S?i�txL�K�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aD�c�R?i�&e��T�?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a}V�6�Q?it�tm�]�?�Unknown
� HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�d`CQ?i�ަ{f�?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������@9������@A������@I������@a*���P?iI��~n�?�Unknown
g"HostStridedSlice"strided_slice(1������@9������@A������@I������@a*���P?i�y�҂v�?�Unknown
e#Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aƦPj��N?i�1�7~�?�Unknown�
�$HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aƦPj��N?i�˱��?�Unknown
�%HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a9%l��M?i����R��?�Unknown
l&HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a9%l��M?i�WF����?�Unknown
V'HostSum"Sum_2(1333333@9333333@A333333@I333333@a9%l��M?i����?�Unknown
�(HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?aw����mG?ii�n.���?�Unknown
�)HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1�������?9�������?A�������?I�������?a�82F?i�������?�Unknown
�*HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1�������?9�������?A�������?I�������?a�HN��C?i���_u��?�Unknown
�+HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�������?9�������?A�������?I�������?a�HN��C?i��d��?�Unknown
�,HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      �?9      �?A      �?I      �?aD�c�B?i�q����?�Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�d`CA?ig���T��?�Unknown
X.HostCast"Cast_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�d`CA?i,�̀���?�Unknown
w/HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�d`CA?i���X���?�Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_2(1�������?9�������?A�������?I�������?a*���@?i��!F���?�Unknown
X1HostEqual"Equal(1333333�?9333333�?A333333�?I333333�?a9%l��=?ie�H���?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1      �?9      �?A      �?I      �?a�!e�8?i_�$u���?�Unknown
`3HostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a�!e�8?i��ȡ���?�Unknown
�4HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a�!e�8?i�m����?�Unknown
V5HostCast"Cast(1�������?9�������?A�������?I�������?a�826?i
4���?�Unknown
�6HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������ @9������ @A�������?I�������?a�826?i-�Qw��?�Unknown
�7HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a�826?iP=��?�Unknown
�8HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�826?is����?�Unknown
X9HostCast"Cast_3(1�������?9�������?A�������?I�������?a�HN��3?iv�r,{��?�Unknown
X:HostCast"Cast_4(1�������?9�������?A�������?I�������?a�HN��3?iy�\����?�Unknown
b;HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a�HN��3?i|zF�i��?�Unknown
�<HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a�HN��3?iC01���?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�d`C1?ib�<�	��?�Unknown
X>HostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?a9%l��-?i$*l���?�Unknown
T?HostMul"Mul(1333333�?9333333�?A333333�?I333333�?a9%l��-?i怛����?�Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?a9%l��-?i��� ���?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a�!e�(?iJ�� ��?�Unknown
sBHostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a�!e�(?i�oM���?�Unknown
wCHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�!e�(?i�0��5��?�Unknown
yDHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�!e�(?i0Nz���?�Unknown
�EHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a�!e�(?i�keK��?�Unknown
�FHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a�!e�(?it������?�Unknown
�GHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�HN��#?i�m,R��?�Unknown
�HHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�HN��#?ixR��L��?�Unknown
aIHostIdentity"Identity(1333333�?9333333�?A333333�?I333333�?a9%l��?i��8�9��?�Unknown�
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a9%l��?i:��~&��?�Unknown
�KHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a9%l��?i�Th?��?�Unknown
�LHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a9%l��?i�������?�Unknown*�H
uHostFlushSummaryWriter"FlushSummaryWriter(1�����	�@9�����	�@A�����	�@I�����	�@al%�2p�?il%�2p�?�Unknown�
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����YY@9�����YY@A�����YY@I�����YY@a�T�/>��?i�:����?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�V@933333�V@A33333�V@I33333�V@a�<�&��?i�C��#�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      S@9      S@A      S@I      S@a� ���ߣ?iĂ݃a�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(133333�J@933333�J@A33333�J@I33333�J@aWì9��?i�����@�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(133333�H@933333�H@A33333�H@I33333�H@a��n!�?i�tb��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(133333�D@933333�D@A33333�D@I33333�D@a�.���?iU��
��?�Unknown
dHostDataset"Iterator::Model(1�����L3@9�����L3@A������,@I������,@a��, ~?i�K��?�Unknown
}	HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      *@9      *@A      *@I      *@a���L2{?i5 ӷ�3�?�Unknown
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      &@9      &@A      &@I      &@a��Vb-w?i�͗�a�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������#@9������#@A������#@I������#@a���>�t?i�1"��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333&@9333333&@A      #@I      #@a� ����s?i����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a�l��r?i�I����?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1������!@9������!@A������!@I������!@a;��{�r?i�@����?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1333333 @9333333 @A333333 @I333333 @a�.Iy�p?iBA3��?�Unknown
iHostWriteSummary"WriteSummary(1       @9       @A       @I       @a)D�~�p?i�3e$A�?�Unknown�
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@aW�d��h?i�	ʬ�Y�?�Unknown
^HostGatherV2"GatherV2(1333333@9333333@A333333@I333333@ao<6��Dh?i@n0r�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@a?w ��e?iV��ڇ�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      @9      @Affffff@Iffffff@aj�_�Ve?i����0��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@aL��ހ�d?i�&�B���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@a���Z��a?iC�'���?�Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @aH7���I]?i_�}T��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      7@9      7@A      @I      @aH7���I]?i{������?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aH7���I]?i��)���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@ay�t*�s\?i�Ǿ����?�Unknown
`HostGatherV2"
GatherV2_1(1������	@9������	@A������	@I������	@a�9�'0�Z?i{��K;�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a>�u%�Y?in}e���?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@ao<6��DX?i��7��#�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a���"JnW?i�I�/�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������@9������@A������@I������@a?w ��U?iuO���:�?�Unknown
g HostStridedSlice"strided_slice(1������@9������@A������@I������@a?w ��U?i�i�cE�?�Unknown
e!Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a3�7���T?i�&9��O�?�Unknown�
�"HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a3�7���T?i���OZ�?�Unknown
�#HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@ad��dT?i��8Zd�?�Unknown
l$HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@ad��dT?i��&�dn�?�Unknown
V%HostSum"Sum_2(1333333@9333333@A333333@I333333@ad��dT?i��5�ox�?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�4s/��O?iZ���b��?�Unknown
�'HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1�������?9�������?A�������?I�������?a��, N?iR����?�Unknown
�(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1�������?9�������?A�������?I�������?a�9�'0�J?i������?�Unknown
�)HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�������?9�������?A�������?I�������?a�9�'0�J?i�� \N��?�Unknown
�*HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      �?9      �?A      �?I      �?a>�u%�I?ih(j���?�Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���"JnG?i��p��?�Unknown
X,HostCast"Cast_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���"JnG?i��{0L��?�Unknown
w-HostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���"JnG?iWa�'��?�Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_2(1�������?9�������?A�������?I�������?a?w ��E?i'�8���?�Unknown
X/HostEqual"Equal(1333333�?9333333�?A333333�?I333333�?ad��dD?i"�ӑ���?�Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      �?9      �?A      �?I      �?a)D�~�@?is;Z�̻�?�Unknown
`1HostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a)D�~�@?i�y�����?�Unknown
�2HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a)D�~�@?i�f�*��?�Unknown
V3HostCast"Cast(1�������?9�������?A�������?I�������?a��, >?i�V,����?�Unknown
�4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������ @9������ @A�������?I�������?a��, >?i������?�Unknown
�5HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a��, >?i����v��?�Unknown
�6HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a��, >?i2}�:��?�Unknown
X7HostCast"Cast_3(1�������?9�������?A�������?I�������?a�9�'0�:?i�0����?�Unknown
X8HostCast"Cast_4(1�������?9�������?A�������?I�������?a�9�'0�:?iS/�����?�Unknown
b9HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a�9�'0�:?i�-��E��?�Unknown
�:HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a�9�'0�:?i�,�����?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a���"Jn7?is��\���?�Unknown
X<HostCast"Cast_1(1333333�?9333333�?A333333�?I333333�?ad��d4?ipJY	��?�Unknown
T=HostMul"Mul(1333333�?9333333�?A333333�?I333333�?ad��d4?im	ݵ���?�Unknown
|>HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?ad��d4?ij�`b��?�Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a)D�~�0?i��#�+��?�Unknown
s@HostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a)D�~�0?i��C��?�Unknown
wAHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a)D�~�0?i�%�[��?�Unknown
yBHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a)D�~�0?iEm�r��?�Unknown
�CHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a)D�~�0?i7d01���?�Unknown
�DHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a)D�~�0?i`������?�Unknown
�EHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a�9�'0�*?i��3N��?�Unknown
�FHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�9�'0�*?i������?�Unknown
aGHostIdentity"Identity(1333333�?9333333�?A333333�?I333333�?ad��d$?i�a:�;��?�Unknown�
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ad��d$?iA|S}��?�Unknown
�IHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ad��d$?i� �����?�Unknown
�JHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ad��d$?i     �?�Unknown