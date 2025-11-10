## Linalg Ops Name Format

some linalg ops contains grad op. currently only unary and binary ops may contain grad op.

Format: `<op_name>_grad(_out)?((_lfs)|(_rfs))?`

`_lfs` or `_rfs` only exists in binary ops.

if the `_out` exists in op name, this means the first input is the output of original op. otherwise, the output isn't exist in the input list.

if the `_lfs` exists in op name, this means the first input is the left hand side of binary op. otherwise, the first input is the right hand side of binary op.

all ops also provide a override function that don't contains unused input such as matrix multiplication.