"�.
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1fffff"�@Afffff"�@a>��?�>�?i>��?�>�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1�����y�@9�����y�@A�����y�@I�����y�@a=R�l��?i>p[�.��?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1������h@9������h@A������h@I������h@a��Z���?i8�zy�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     0`@9     0`@A     0`@I     0`@aL�oӭA�?i��=_�}�?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(133333�Q@933333�Q@A33333�Q@I33333�Q@a���٪�?i�`���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffffP@9fffffP@AfffffP@IfffffP@a������?i�Fd쉍�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     @F@9     @F@A     @F@I     @F@a�u��LX�?i}x����?�Unknown
o	Host_FusedMatMul"sequential/dense/Relu(1�����L=@9�����L=@A�����L=@I�����L=@a��;��l}?i��"��!�?�Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(13333336@93333336@A3333336@I3333336@a��rKv?i��:�[N�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������0@9������0@A������0@I������0@a�Y��%�p?i�)��p�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������,@9������,@A������,@I������,@a���@�l?if+�)��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff,@9ffffff,@Affffff,@Iffffff,@a�Ӱj�l?i�������?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      *@9      *@A      *@I      *@a�Gej?i������?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1ffffff(@9ffffff(@Affffff(@Iffffff(@a�W�h?i��6)��?�Unknown
iHostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@aNLf��f?i ����?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333*@9333333*@A������$@I������$@aK�M�d?i�[�Ҿ�?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1������"@9������"@A������"@I������"@a�uf��b?i'¦�l�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a��ݙ�^?i�����)�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@aS�H��P[?i�Ó�X7�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@a܆aϸNY?i�t�Y D�?�Unknown
dHostDataset"Iterator::Model(1333333%@9333333%@A      @I      @a���6X?i�uP�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @a���6X?isK�\�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������ @9������ @A������@I������@a~���_�W?i��@�g�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a5�*�~V?i����3s�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@a��d9��T?ia�p�}�?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a��6��|T?iP�����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@a`�}U�R?i;<rT��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@aGVO�~zR?i�b�����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a��9gyP?i��Wn���?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a��ݙ�N?icn�T<��?�Unknown
e Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a�@�t�L?i����C��?�Unknown�
s!HostCast"!sequential/dropout_1/dropout/Cast(1333333@9333333@A333333@I333333@aS�H��PK?i��3��?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a�ŏ��I?iՍ�����?�Unknown
g#HostStridedSlice"strided_slice(1������	@9������	@A������	@I������	@a�ŏ��I?i�qq����?�Unknown
�$HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�G3���H?i�>�,��?�Unknown
�%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a�G3���H?ij��f��?�Unknown
�&HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a���6H?i�Azm��?�Unknown
�'HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @a���6H?i�v�t��?�Unknown
�(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a5�*�~F?iA����?�Unknown
o)HostMul"sequential/dropout/dropout/Mul(1ffffff@9ffffff@Affffff@Iffffff@a5�*�~F?i��]v���?�Unknown
x*HostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff9@9ffffff9@A������@I������@a��d9��D?i��kW���?�Unknown
�+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a�R��D?i� \����?�Unknown
l,HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a�R��D?icLC���?�Unknown
`-HostGatherV2"
GatherV2_1(1ffffff@9ffffff@Affffff@Iffffff@aGVO�~zB?i�v ���?�Unknown
�.HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff#@9ffffff#@A�������?I�������?a��d9��4?i���S2��?�Unknown
a/HostIdentity"Identity(1�������?9�������?A�������?I�������?a�ŏ��?i      �?�Unknown�*�-
uHostFlushSummaryWriter"FlushSummaryWriter(1�����y�@9�����y�@A�����y�@I�����y�@ad�;z8��?id�;z8��?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1������h@9������h@A������h@I������h@a�W�'�?iV֬]T�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1     0`@9     0`@A     0`@I     0`@a��H)ʎ�?ia_A �?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(133333�Q@933333�Q@A33333�Q@I33333�Q@a ��pX�?iڅ���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1fffffP@9fffffP@AfffffP@IfffffP@aC�=+�G�?i4G	��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1     @F@9     @F@A     @F@I     @F@a	�2+w�?i��� �q�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����L=@9�����L=@A�����L=@I�����L=@a�������?i
��O��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(13333336@93333336@A3333336@I3333336@a�El��?i"�y*� �?�Unknown
�	HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������0@9������0@A������0@I������0@a\�II��|?i�6�Z�?�Unknown
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������,@9������,@A������,@I������,@a��>ăx?iKډ'!��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff,@9ffffff,@Affffff,@Iffffff,@aV��h�,x?i8[\z��?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      *@9      *@A      *@I      *@a�`c�!v?i��!����?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1ffffff(@9ffffff(@Affffff(@Iffffff(@a��
��t?i �6�G�?�Unknown
iHostWriteSummary"WriteSummary(1������&@9������&@A������&@I������&@a�N��Phs?i�h�,8�?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1333333*@9333333*@A������$@I������$@aσ���q?i���*[�?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1������"@9������"@A������"@I������"@a9"$�2�o?iȿ�4�z�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1ffffff@9ffffff@Affffff@Iffffff@a
i��k�i?i1d2����?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a����'g?ic�۫�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@ar��Kse?i�Z�O��?�Unknown
dHostDataset"Iterator::Model(1333333%@9333333%@A      @I      @af`Y4�md?i�׼��?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @af`Y4�md?iH8�*��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������ @9������ @A������@I������@aܯ$^�d?i�1�IA��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1ffffff@9ffffff@Affffff@Iffffff@a=���&c?i��qpR�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1������@9������@A������@I������@aܳ��a?irl��#�?�Unknown
^HostGatherV2"GatherV2(1ffffff@9ffffff@Affffff@Iffffff@a�+�U]a?i��Ed4�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@aai�S� `?i���dD�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@a�q��S_?i�rxT�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a3+w�=[?iVK��a�?�Unknown
ZHostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a
i��k�Y?i��7�n�?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a�Eh�p�W?i�Q��z�?�Unknown�
sHostCast"!sequential/dropout_1/dropout/Cast(1333333@9333333@A333333@I333333@a����'W?i Qr~��?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a�",�u�U?i1�8� ��?�Unknown
g!HostStridedSlice"strided_slice(1������	@9������	@A������	@I������	@a�",�u�U?iB}����?�Unknown
�"HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a{���!U?i���t��?�Unknown
�#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a{���!U?i@���?�Unknown
�$HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @af`Y4�mT?i�l��8��?�Unknown
�%HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @af`Y4�mT?id��o��?�Unknown
�&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a=���&S?i�\�w���?�Unknown
o'HostMul"sequential/dropout/dropout/Mul(1ffffff@9ffffff@Affffff@Iffffff@a=���&S?i �
���?�Unknown
x(HostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff9@9ffffff9@A������@I������@aܳ��Q?i�y�J[��?�Unknown
�)HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a {J�+Q?i.��`���?�Unknown
l*HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a {J�+Q?ilćva��?�Unknown
`+HostGatherV2"
GatherV2_1(1ffffff@9ffffff@Affffff@Iffffff@a�q��SO?iH��86��?�Unknown
�,HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff#@9ffffff#@A�������?I�������?aܳ��A?i?-�X���?�Unknown
a-HostIdentity"Identity(1�������?9�������?A�������?I�������?a�",�u�%?i     �?�Unknown�