"�W
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1fffff��@Afffff��@a��K�R�?i��K�R�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff�@9fffff�@Afffff�@Ifffff�@a]���?i��q�4�?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1������i@9������i@A������i@I������i@a3�����?i*����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1����̌Z@9����̌Z@A����̌Z@I����̌Z@aTz�9Ɓ�?i�#��"��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����LW@9�����LW@A�����LW@I�����LW@a;a	vb�?i�r6<�?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1      Q@9      Q@A      Q@I      Q@a�U�s�?iǻe�Կ�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �K@9     �K@A     �K@I     �K@a����稊?i�G|2x*�?�Unknown
q	Host_FusedMatMul"sequential/dense_1/Relu(1�����E@9�����E@A�����E@I�����E@a�3cE�?i*�H��{�?�Unknown
o
Host_FusedMatMul"sequential/dense/Relu(1������>@9������>@A������>@I������>@a��&r�e}?iU�,nY��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     �<@9     �<@A     �<@I     �<@a��#�\a{?i5G�'��?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1     �4@9     �4@A     �4@I     �4@ad�mұs?i�z���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(13333332@93333332@A3333332@I3333332@a��v'|q?iΨmx7�?�Unknown
oHostMul"sequential/dropout/dropout/Mul(13333331@93333331@A3333331@I3333331@a'�,6�p?i.�Ň�X�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      -@9      -@A      -@I      -@a0�$bU�k?i��'�`t�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff)@9ffffff)@Affffff)@Iffffff)@a ��fh?i���ǌ�?�Unknown
dHostDataset"Iterator::Model(1ffffff/@9ffffff/@A������(@I������(@a��/�g?i�7��i��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������+@9������+@A������&@I������&@aտd}�e?i�T {Q��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff$@9ffffff$@Affffff$@Iffffff$@a�L:�c?iDnL����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff'@9ffffff'@A������#@I������#@a�Dy�b?i���.���?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@ad�4�Ja?i���%
��?�Unknown
iHostWriteSummary"WriteSummary(1������ @9������ @A������ @I������ @a	/��#`?iܲl�-�?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�I!�\Y?i��pl��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@anE�_�W?i$�l��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a�l>�V?i�;&�?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@aտd}�U?ie�T�1�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @a��\�"U?iD��X�;�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a�5�rR?i_
�d�D�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@aYW'8h�M?i5�XL�?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@a�R%(�oL?i�{�tS�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@a&L"�!J?i&?!�Y�?�Unknown
s HostCast"!sequential/dropout_1/dropout/Cast(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�I!�\I?io.�YT`�?�Unknown
`!HostGatherV2"
GatherV2_1(1������	@9������	@A������	@I������	@a�G  !�H?i�6bzf�?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a�G  !�H?i�>Aj�l�?�Unknown
�#HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1������	@9������	@A������	@I������	@a�G  !�H?i�F�r�r�?�Unknown
�$HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������	@9������	@A������	@I������	@a�G  !�H?i�N�z�x�?�Unknown
V%HostSum"Sum_2(1      @9      @A      @I      @a1C�G?iHV}"�~�?�Unknown
�&HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a�@��IF?i�]��B��?�Unknown
x'HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������=@9������=@Affffff@Iffffff@a�>��E?i�d/ᣉ�?�Unknown
�(HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�>��E?i�kg(��?�Unknown
e)Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@ay<�[�D?i�r]?5��?�Unknown�
�*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@a;:К�C?iy&4��?�Unknown
g+HostStridedSlice"strided_slice(1������@9������@A������@I������@a;:К�C?i��3��?�Unknown
X,HostEqual"Equal(1      @9      @A      @I      @a�7��6C?i�7� ��?�Unknown
l-HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a�7��6C?iA��yΧ�?�Unknown
�.HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @a�7��6C?i��0���?�Unknown
�/HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a�5�rB?i��K�8��?�Unknown
X0HostCast"Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a�3�W�A?ii�9���?�Unknown
�1HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1       @9       @A       @I       @a�Y(@)�>?it�a�{��?�Unknown
�2HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff�?9ffffff�?Affffff�?Iffffff�?aU&0�4=?i?�Gf"��?�Unknown
q3HostMul" sequential/dropout/dropout/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?aU&0�4=?i
�-����?�Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_2(1333333�?9333333�?A333333�?I333333�?a&L"�!:?iT��/��?�Unknown
�5HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333�?9333333�?A333333�?I333333�?a&L"�!:?i���cQ��?�Unknown
�6HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333�?9333333�?A333333�?I333333�?a&L"�!:?i�S����?�Unknown
7HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      �?9      �?A      �?I      �?a1C�7?i��1lw��?�Unknown
�8HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      �?9      �?A      �?I      �?a1C�7?ix�@Y��?�Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?a;:К�3?i��i����?�Unknown
�:HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333.@9333333.@A�������?I�������?a;:К�3?i��&X��?�Unknown
�;HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a;:К�3?iM�����?�Unknown
}<HostMul",gradient_tape/sequential/dropout/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a�5�r2?iT�5�%��?�Unknown
q=HostMul" sequential/dropout_1/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a�5�r2?i[�M t��?�Unknown
�>HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1333333�?9333333�?A333333�?I333333�?a�5�r2?ib�ec���?�Unknown
�?HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?aF1���0?i(�;v���?�Unknown
V@HostCast"Cast(1      �?9      �?A      �?I      �?a�Y(@)�.?i���X���?�Unknown
XAHostCast"Cast_2(1      �?9      �?A      �?I      �?a�Y(@)�.?i4�c;���?�Unknown
�BHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      �?9      �?A      �?I      �?a�Y(@)�.?i������?�Unknown
bCHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a�P$ %�+?i��I�]��?�Unknown
wDHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�P$ %�+?iD㛂��?�Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a�G  !�(?iH����?�Unknown
FHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1�������?9�������?A�������?I�������?a�G  !�(?iL细+��?�Unknown
XGHostCast"Cast_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i�؃��?�Unknown
|HHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i��W*���?�Unknown
`IHostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i��%|4��?�Unknown
�JHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i\��͌��?�Unknown
�KHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i �����?�Unknown
�LHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�>��%?i��q=��?�Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a�Y(@)�?i'��b3��?�Unknown
XNHostCast"Cast_4(1      �?9      �?A      �?I      �?a�Y(@)�?ij�#T)��?�Unknown
TOHostMul"Mul(1      �?9      �?A      �?I      �?a�Y(@)�?i��mE��?�Unknown
sPHostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a�Y(@)�?i���6��?�Unknown
sQHostMul""sequential/dropout_1/dropout/Mul_1(1      �?9      �?A      �?I      �?a�Y(@)�?i3�(��?�Unknown
vRHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a�G  !�?i5�	����?�Unknown
aSHostIdentity"Identity(1�������?9�������?A�������?I�������?a�G  !�?i7�����?�Unknown�
wTHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�G  !�?i9�kY��?�Unknown
yUHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a�G  !�?i;�!,��?�Unknown
�VHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�G  !�?i=�)����?�Unknown
�WHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�5�r?i���}v��?�Unknown
�XHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�5�r?i���
��?�Unknown
�YHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�5�r?i��{����?�Unknown
uZHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a�G  !�?i     �?�Unknown*�V
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff�@9fffff�@Afffff�@Ifffff�@a�L��4�?i�L��4�?�Unknown�
�HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1������i@9������i@A������i@I������i@aq�(�,P�?i"b�l: �?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1����̌Z@9����̌Z@A����̌Z@I����̌Z@a����<�?i�?6��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����LW@9�����LW@A�����LW@I�����LW@a�X�+DE�?i�i�x[��?�Unknown
�HostRandomUniform"9sequential/dropout_1/dropout/random_uniform/RandomUniform(1      Q@9      Q@A      Q@I      Q@aT���ND�?i��?�}��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �K@9     �K@A     �K@I     �K@a.��=U�?i����'��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1�����E@9�����E@A�����E@I�����E@a�Z�C�?i�ζ�@7�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1������>@9������>@A������>@I������>@ad��>K�?i�~����?�Unknown
y	HostMatMul"%gradient_tape/sequential/dense/MatMul(1     �<@9     �<@A     �<@I     �<@a��ӄ�?iM�?�Unknown
o
HostSoftmax"sequential/dense_2/Softmax(1     �4@9     �4@A     �4@I     �4@a�6oG涂?i�	��Y�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(13333332@93333332@A3333332@I3333332@a[�Kb��?iM�N9a��?�Unknown
oHostMul"sequential/dropout/dropout/Mul(13333331@93333331@A3333331@I3333331@a"���\g?i�*8�/��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      -@9      -@A      -@I      -@a �w�^yz?iC��"�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff)@9ffffff)@Affffff)@Iffffff)@aɱ�
0w?i���ł>�?�Unknown
dHostDataset"Iterator::Model(1ffffff/@9ffffff/@A������(@I������(@a(��Uuv?i�p��lk�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������+@9������+@A������&@I������&@a|�Bj�t?i]���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff$@9ffffff$@Affffff$@Iffffff$@a��G��r?i��L��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff'@9ffffff'@A������#@I������#@a��>��q?i�����?�Unknown
qHostCast"sequential/dropout/dropout/Cast(1      "@9      "@A      "@I      "@a�ؙ��np?i�<>���?�Unknown
iHostWriteSummary"WriteSummary(1������ @9������ @A������ @I������ @a����f�n?i%µ���?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�={h�h?ic=c�5�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a�*�ѣf?i�D5]L�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������@9������@A������@I������@a0�`�e?i�t���a�?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@a|�Bj�d?i�j���v�?�Unknown
�HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1      @9      @A      @I      @a��td?i�&�sΊ�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a%�p�a?i���U��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@a�w��{\?ih��K���?�Unknown
ZHostArgMax"ArgMax(1������@9������@A������@I������@aZdcm�[?i5���?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333@9333333@A333333@I333333@auG����X?i��q���?�Unknown
sHostCast"!sequential/dropout_1/dropout/Cast(1ffffff
@9ffffff
@Affffff
@Iffffff
@a�={h�X?i]M"Q���?�Unknown
`HostGatherV2"
GatherV2_1(1������	@9������	@A������	@I������	@a24A'�^W?i��5�<��?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������	@9������	@A������	@I������	@a24A'�^W?i��I���?�Unknown
�!HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1������	@9������	@A������	@I������	@a24A'�^W?i+/]}���?�Unknown
�"HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������	@9������	@A������	@I������	@a24A'�^W?i��p�J��?�Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a� ͤ��U?iU6CO?
�?�Unknown
�$HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@aM�c�-U?i���A��?�Unknown
x%HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������=@9������=@Affffff@Iffffff@a�Y"�rT?ih,���?�Unknown
�&HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�Y"�rT?i�X1I)�?�Unknown
e'Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a
���S?iq�-%3�?�Unknown�
�(HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������@9������@A������@I������@ai���R?i��׮�<�?�Unknown
g)HostStridedSlice"strided_slice(1������@9������@A������@I������@ai���R?ik�'0"F�?�Unknown
X*HostEqual"Equal(1      @9      @A      @I      @a��^BR?i�"W6CO�?�Unknown
l+HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a��^BR?i[x�<dX�?�Unknown
�,HostGreaterEqual")sequential/dropout_1/dropout/GreaterEqual(1      @9      @A      @I      @a��^BR?i�͵B�a�?�Unknown
�-HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a%�p�Q?iG���Hj�?�Unknown
X.HostCast"Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a��6��P?i���ݮr�?�Unknown
�/HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1       @9       @A       @I       @a>�1z6M?i�>|�y�?�Unknown
�0HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�m����K?iq�����?�Unknown
q1HostMul" sequential/dropout/dropout/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�m����K?i�4�܇�?�Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_2(1333333�?9333333�?A333333�?I333333�?auG����H?i�@���?�Unknown
�3HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333�?9333333�?A333333�?I333333�?auG����H?ipkG��?�Unknown
�4HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333�?9333333�?A333333�?I333333�?auG����H?i�|�J|��?�Unknown
5HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1      �?9      �?A      �?I      �?a� ͤ��E?i
�~����?�Unknown
�6HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      �?9      �?A      �?I      �?a� ͤ��E?iR�g�p��?�Unknown
t7HostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?ai���B?i���/��?�Unknown
�8HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333.@9333333.@A�������?I�������?ai���B?i�շ9��?�Unknown
�9HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?ai���B?i�_z���?�Unknown
}:HostMul",gradient_tape/sequential/dropout/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a%�p�A?iI+�?��?�Unknown
q;HostMul" sequential/dropout_1/dropout/Mul(1333333�?9333333�?A333333�?I333333�?a%�p�A?i��nr��?�Unknown
�<HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1333333�?9333333�?A333333�?I333333�?a%�p�A?i�������?�Unknown
�=HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a����)@?i�\���?�Unknown
V>HostCast"Cast(1      �?9      �?A      �?I      �?a>�1z6=?i"Ţ�~��?�Unknown
X?HostCast"Cast_2(1      �?9      �?A      �?I      �?a>�1z6=?iR��%��?�Unknown
�@HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1      �?9      �?A      �?I      �?a>�1z6=?i�	/����?�Unknown
bAHostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?a�Z),�J:?i��T���?�Unknown
wBHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�Z),�J:?i�z+_��?�Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a24A'�^7?i��~K��?�Unknown
DHostMul".gradient_tape/sequential/dropout_1/dropout/Mul(1�������?9�������?A�������?I�������?a24A'�^7?i&��6��?�Unknown
XEHostCast"Cast_3(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?iH/h;���?�Unknown
|FHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?ijzL�S��?�Unknown
`GHostDivNoNan"
div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?i��0����?�Unknown
�HHostMul"0gradient_tape/sequential/dropout_1/dropout/Mul_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?i�Up��?�Unknown
�IHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?i�[�����?�Unknown
�JHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�Y"�r4?i�����?�Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a>�1z6-?i
��x`��?�Unknown
XLHostCast"Cast_4(1      �?9      �?A      �?I      �?a>�1z6-?i"�#�3��?�Unknown
TMHostMul"Mul(1      �?9      �?A      �?I      �?a>�1z6-?i:��G��?�Unknown
sNHostReadVariableOp"SGD/Cast/ReadVariableOp(1      �?9      �?A      �?I      �?a>�1z6-?iR�i����?�Unknown
sOHostMul""sequential/dropout_1/dropout/Mul_1(1      �?9      �?A      �?I      �?a>�1z6-?ij����?�Unknown
vPHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?a24A'�^'?i}p�$��?�Unknown
aQHostIdentity"Identity(1�������?9�������?A�������?I�������?a24A'�^'?i�����?�Unknown�
wRHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a24A'�^'?i�X����?�Unknown
ySHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1�������?9�������?A�������?I�������?a24A'�^'?i��Ʌ��?�Unknown
�THostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a24A'�^'?i�@�����?�Unknown
�UHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a%�p�!?i��&��?�Unknown
�VHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a%�p�!?i��\�,��?�Unknown
�WHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a%�p�!?i�ž	E��?�Unknown
uXHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a24A'�^?i�������?�Unknown