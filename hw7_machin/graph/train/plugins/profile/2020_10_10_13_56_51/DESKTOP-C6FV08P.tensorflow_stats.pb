"�W
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1�����k�@9�����k�@A�����k�@I�����k�@a�Z`)��?i�Z`)��?�Unknown�
BHostIDLE"IDLE1�����$�@A�����$�@a�[��T�?i[��	��?�Unknown
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     �j@9     �j@A     �j@I     �j@a���b��?i� P��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffff&^@9fffff&^@Afffff&^@Ifffff&^@a�B� -�?i/'��w�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�Y@933333�Y@A33333�Y@I33333�Y@a��k��?i=�P�@�?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1fffff�P@9fffff�P@Afffff�P@Ifffff�P@aL��n���?i�l�����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffffH@9ffffffH@AffffffH@IffffffH@af��J���?i(�+��?�Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(133333sE@933333sE@A33333sE@I33333sE@a�B�����?i3쬗r�?�Unknown
o
Host_FusedMatMul"sequential/dense/Relu(133333�@@933333�@@A33333�@@I33333�@@a��*zg�?i�բ��?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1      9@9      9@A      9@I      9@a `�M1x?i�V~q��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1������5@9������5@A������5@I������5@a�aa��t?i�9y��?�Unknown
oHostMul"sequential/dropout/dropout/Mul(1ffffff/@9ffffff/@Affffff/@Iffffff/@afD�[�bn?i��C6,�?�Unknown
dHostDataset"Iterator::Model(1������2@9������2@A������+@I������+@a�^�a��j?id��7G�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������+@9������+@A������+@I������+@a�^�a��j?i�SX,b�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff+@9ffffff+@Affffff+@Iffffff+@af2�܃j?iυ��|�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1������)@9������)@A������)@I������)@a�����h?ii���M��?�Unknown
`HostDivNoNan"
div_no_nan(1333333(@9333333(@A333333(@I333333(@a3yogkg?i�	����?�Unknown
iHostWriteSummary"WriteSummary(1ffffff&@9ffffff&@Affffff&@Iffffff&@afRj3�e?i�[cLf��?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      )@9      )@A������$@I������$@aͼpP� d?i�̳ ���?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@a |&�-ka?i!�N���?�Unknown
^HostGatherV2"GatherV2(1ffffff!@9ffffff!@Affffff!@Iffffff!@af r9��`?i!e�����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a�����X?ino��+�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a����z�V?i�L���?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1������@9������@A������@I������@a����z�V?il*]L�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a�aa��T?i�K�u&�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      @9      @A333333@I333333@a3s�6u�R?i�"��/�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1333333@9333333@A333333@I333333@a3s�6u�R?i�j�C
9�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a� p]1R?i!v:�"B�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@a� p]1R?i���;K�?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a |&�-kQ?i��7�S�?�Unknown
x HostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff9@9ffffff9@A������@I������@a�)�Q?il�Bu\�?�Unknown
�!HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff@9ffffff@Affffff@Iffffff@a�d����O?i�W�dd�?�Unknown
Z"HostArgMax"ArgMax(1      @9      @A      @I      @a ��n�N?iu؅"l�?�Unknown
s#HostCast"!sequential/dropout_1/dropout/Cast(1ffffff
@9ffffff
@Affffff
@Iffffff
@af>+!�I?i]6#��r�?�Unknown
�$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a�����H?i���
�x�?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@a��#��G?i���~�?�Unknown
�&HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a��#��G?i}M�분�?�Unknown
�'HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a��#��G?iz�ܶ��?�Unknown
�(HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a3�B�bsF?i%gմS��?�Unknown
�)HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1������@9������@A������@I������@a�aa��D?i}��u���?�Unknown
l*HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a�aa��D?i��6ǚ�?�Unknown
�+HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������@9������@A������@I������@aͼpP� D?i4�kϟ�?�Unknown
g,HostStridedSlice"strided_slice(1������@9������@A������@I������@aͼpP� D?i3P�פ�?�Unknown
e-Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a �äZC?i90J���?�Unknown�
�.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������ @9�������?A������ @I�������?ä́���A@?i��þ��?�Unknown
�/HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������ @9������ @A������ @I������ @ä́���A@?i�e=ϱ�?�Unknown
}0HostMul",gradient_tape/sequential/dropout/dropout/Mul(1������ @9������ @A������ @I������ @ä́���A@?i\��ߵ�?�Unknown
�1HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1������ @9������ @A������ @I������ @ä́���A@?i���0��?�Unknown
�2HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1�������?9�������?A�������?I�������?a�,�ѯ�;?i�(�l��?�Unknown
3HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a3���PR:?i_'�P���?�Unknown
�4HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a3���PR:?i&՚ ��?�Unknown
q5HostMul" sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a3���PR:?i�$��J��?�Unknown
V6HostSum"Sum_2(1�������?9�������?A�������?I�������?a�����8?ij��c��?�Unknown
�7HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�������?9�������?A�������?I�������?a�����8?i��Sa|��?�Unknown
8HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      �?9      �?A      �?I      �?a P3��97?ig0��c��?�Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?afRj3�5?i�z:��?�Unknown
�:HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff�?9ffffff�?Affffff�?Iffffff�?afRj3�5?i��~����?�Unknown
�;HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aͼpP� 4?i��R��?�Unknown
t<HostAssignAddVariableOp"AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?a3s�6u�2?i勞����?�Unknown
X=HostCast"Cast_2(1�������?9�������?A�������?I�������?a�)�1?i�:s����?�Unknown
�>HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a�)�1?iy�6����?�Unknown
q?HostMul" sequential/dropout_1/dropout/Mul(1�������?9�������?A�������?I�������?a�)�1?i>f����?�Unknown
�@HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a�)�1?i���)��?�Unknown
XAHostEqual"Equal(1      �?9      �?A      �?I      �?a ��n�.?i�U���?�Unknown
�BHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      !@9      !@A      �?I      �?a ��n�.?i;�~���?�Unknown
VCHostCast"Cast(1�������?9�������?A�������?I�������?a�,�ѯ�+?i��{m���?�Unknown
XDHostCast"Cast_3(1�������?9�������?A�������?I�������?a�,�ѯ�+?i!�xX���?�Unknown
�EHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�,�ѯ�+?i�vCB��?�Unknown
|FHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1�������?9�������?A�������?I�������?a�����(?i�菢���?�Unknown
�GHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1�������?9�������?A�������?I�������?a�����(?i(ʩ[��?�Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?afRj3�%?iHo�Ե��?�Unknown
XIHostCast"Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?afRj3�%?ih���?�Unknown
bJHostDivNoNan"div_no_nan_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?afRj3�%?i��M{k��?�Unknown
TKHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a3s�6u�"?i"���?�Unknown
sLHostMul""sequential/dropout_1/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a3s�6u�"?iv��	���?�Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a ��n�?iD�dŵ��?�Unknown
XNHostCast"Cast_4(1      �?9      �?A      �?I      �?a ��n�?i�Ԁ���?�Unknown
�OHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a ��n�?i�E<���?�Unknown
�PHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a ��n�?i�>�����?�Unknown
aQHostIdentity"Identity(1�������?9�������?A�������?I�������?a�����?iS/B'c��?�Unknown�
sRHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a�����?i��V)��?�Unknown
wSHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�����?i�\����?�Unknown
yTHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�����?iB鵵��?�Unknown
�UHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�����?i��u�{��?�Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a3s�6u�?ic����?�Unknown
�WHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a3s�6u�?i�Z�,���?�Unknown
�XHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a3s�6u�?i[s�9��?�Unknown
uYHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�����?i��9���?�Unknown
wZHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�����?i�������?�Unknown*�V
uHostFlushSummaryWriter"FlushSummaryWriter(1�����k�@9�����k�@A�����k�@I�����k�@aP�u���?iP�u���?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1     �j@9     �j@A     �j@I     �j@ar&T��?iGXt7�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffff&^@9fffff&^@Afffff&^@Ifffff&^@a����C��?i��E���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(133333�Y@933333�Y@A33333�Y@I33333�Y@a� ��A�?i�b'��?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1fffff�P@9fffff�P@Afffff�P@Ifffff�P@a�&LL�̙?i;&v܍��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1ffffffH@9ffffffH@AffffffH@IffffffH@a	�V+�?i���4�Z�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(133333sE@933333sE@A33333sE@I33333sE@aA{����?iմB���?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(133333�@@933333�@@A33333�@@I33333�@@a$, �v�?i�d���J�?�Unknown
o	HostSoftmax"sequential/dense_2/Softmax(1      9@9      9@A      9@I      9@a 1���?iF)NBϘ�?�Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1������5@9������5@A������5@I������5@aG����܀?i��ȰA��?�Unknown
oHostMul"sequential/dropout/dropout/Mul(1ffffff/@9ffffff/@Affffff/@Iffffff/@an���x?ij��G�?�Unknown
dHostDataset"Iterator::Model(1������2@9������2@A������+@I������+@a�R�\��u?i���8�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������+@9������+@A������+@I������+@a�R�\��u?i�0�]d�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff+@9ffffff+@Affffff+@Iffffff+@a}%�q�cu?iec�ݎ�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1������)@9������)@A������)@I������)@a8Y���s?i�{�ն�?�Unknown
`HostDivNoNan"
div_no_nan(1333333(@9333333(@A333333(@I333333(@aV��3�r?i(�0���?�Unknown
iHostWriteSummary"WriteSummary(1ffffff&@9ffffff&@Affffff&@Iffffff&@a�)u{|q?iA���?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      )@9      )@A������$@I������$@a}89Ȼ<p?iu��� �?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@av�'�Xl?ij�^�*<�?�Unknown
^HostGatherV2"GatherV2(1ffffff!@9ffffff!@Affffff!@Iffffff!@aGms�*k?i�NPnUW�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a8Y���c?i0ZiQk�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a@v�6Klb?i�8V��}�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1������@9������@A������@I������@a@v�6Klb?i��)��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@aG�����`?i��+���?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      @9      @A333333@I333333@a��6��]?i2G���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1333333@9333333@A333333@I333333@a��6��]?i�Yb� ��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a
�`Z]?if�����?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@a
�`Z]?iar«Z��?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @av�'�X\?i\�g��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff9@9ffffff9@A������@I������@a����xz[?i)^�%��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff@9ffffff@Affffff@Iffffff@aO�FYٚY?in7���?�Unknown
Z HostArgMax"ArgMax(1      @9      @A      @I      @a�/΂��X?i�h��o�?�Unknown
s!HostCast"!sequential/dropout_1/dropout/Cast(1ffffff
@9ffffff
@Affffff
@Iffffff
@a���ڛT?i`*K��?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a8Y���S?i���%�?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@an���\S?i��.�i/�?�Unknown
�$HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������@9������@A������@I������@an���\S?iC�9�?�Unknown
�%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@an���\S?i��'�B�?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�H�K[R?i�]�>�K�?�Unknown
�'HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1������@9������@A������@I������@aG�����P?ix���BT�?�Unknown
l(HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@aG�����P?iBlڰ\�?�Unknown
�)HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������@9������@A������@I������@a}89Ȼ<P?i�+P8�d�?�Unknown
g*HostStridedSlice"strided_slice(1������@9������@A������@I������@a}89Ȼ<P?izH4��l�?�Unknown
e+Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @ag���9O?i�(-�t�?�Unknown�
�,HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������ @9�������?A������ @I�������?a�/�:J?i�y�J{�?�Unknown
�-HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������ @9������ @A������ @I������ @a�/�:J?i[�`ف�?�Unknown
}.HostMul",gradient_tape/sequential/dropout/dropout/Mul(1������ @9������ @A������ @I������ @a�/�:J?i�h��?�Unknown
�/HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1������ @9������ @A������ @I������ @a�/�:J?i��\����?�Unknown
�0HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1�������?9�������?A�������?I�������?a_��(z{F?i�"盕��?�Unknown
1HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a��{�;E?i"����?�Unknown
�2HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��{�;E?i!%y3��?�Unknown
q3HostMul" sequential/dropout/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a��{�;E?i
 �g���?�Unknown
V4HostSum"Sum_2(1�������?9�������?A�������?I�������?a8Y���C?i��wf���?�Unknown
�5HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�������?9�������?A�������?I�������?a8Y���C?i��+e���?�Unknown
6HostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1      �?9      �?A      �?I      �?a��";�B?i_,�s/��?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�)u{|A?i�vђ���?�Unknown
�8HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�)u{|A?iW������?�Unknown
�9HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a}89Ȼ<@?i�Ϡ����?�Unknown
t:HostAssignAddVariableOp"AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?a��6��=?iơ����?�Unknown
X;HostCast"Cast_2(1�������?9�������?A�������?I�������?a����xz;?i�7�n+��?�Unknown
�<HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a����xz;?i��޽���?�Unknown
q=HostMul" sequential/dropout_1/dropout/Mul(1�������?9�������?A�������?I�������?a����xz;?i�c�
��?�Unknown
�>HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1�������?9�������?A�������?I�������?a����xz;?i��\y��?�Unknown
X?HostEqual"Equal(1      �?9      �?A      �?I      �?a�/΂��8?iXSF����?�Unknown
�@HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      !@9      !@A      �?I      �?a�/΂��8?i�v���?�Unknown
VAHostCast"Cast(1�������?9�������?A�������?I�������?a_��(z{6?i�ʻ����?�Unknown
XBHostCast"Cast_3(1�������?9�������?A�������?I�������?a_��(z{6?iP� �V��?�Unknown
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a_��(z{6?i�Fh&��?�Unknown
|DHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1�������?9�������?A�������?I�������?a8Y���3?iT����?�Unknown
�EHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1�������?9�������?A�������?I�������?a8Y���3?i���f%��?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_4(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�)u{|1?i�mh�T��?�Unknown
XGHostCast"Cast_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�)u{|1?i;ׅ���?�Unknown
bHHostDivNoNan"div_no_nan_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�)u{|1?iy�E���?�Unknown
TIHostMul"Mul(1333333�?9333333�?A333333�?I333333�?a��6��-?i�!ɴ���?�Unknown
sJHostMul""sequential/dropout_1/dropout/Mul_1(1333333�?9333333�?A333333�?I333333�?a��6��-?i��LTs��?�Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a�/΂��(?i|����?�Unknown
XLHostCast"Cast_4(1      �?9      �?A      �?I      �?a�/΂��(?i_�|����?�Unknown
�MHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a�/΂��(?iBc"��?�Unknown
�NHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a�/΂��(?i%>����?�Unknown
aOHostIdentity"Identity(1�������?9�������?A�������?I�������?a8Y���#?i�.Z����?�Unknown�
sPHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?a8Y���#?i��1��?�Unknown
wQHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a8Y���#?iG�Qq��?�Unknown
yRHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a8Y���#?i� a���?�Unknown
�SHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a8Y���#?i������?�Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1333333�?9333333�?A333333�?I333333�?a��6��?i;�Ϡ���?�Unknown
�UHostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��6��?i�Z�p���?�Unknown
�VHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��6��?iKS@���?�Unknown
uWHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a8Y���?i��) `��?�Unknown
wXHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a8Y���?i      �?�Unknown