## 64bit Linear Hash Design In Apple Silicon

Currently native atomicCAS 64bit isn't supported in apple metal compute shader. To support 64bit (only 62bit key available for user) hash table, we design a 32bit atomicCAS based method.

### Data Format

We only support 62bit `uint64_t` hash key, which means the most two bit of `uint64_t` is reserved and can't be used by user. 

### Empty Key

Unlike CUDA implementation, the empty key of 64bit hash in apple metal is always `0xffffffffffffffff` and can't be set by user.

### Hash Key Storage

Since only 32bit atomicCAS available, we must make sure the most significant word and the last word has a empty key (`0xffffffff` in impl). Recall we only allow 62bit for user, we convert user provided key to following format. We use 16 bit as example

* user key format: `0b **** **** **** ****`

* user key format masked: `0b 00** **** **** ****`

* user key converted: `0b *0** **** 0*** ****`

In format above, we first reset reserved two bits, then swap the first bit of most word with the first bit of last world (byte in example).

This format make sure the first and the last word of a 64bit key always isn't equal to the empty key `0xffffffff`.

### The Algorithm

To simulate 64bit atomicCAS process, we design following algo:

1. do 32bit atomicCAS on the most significant word until success or the current most word not equal to empty key.

After this step, only one thread will success, the hash slot is owned by this thread.

2. Succeed thread save the last word of key to hash key storage.

3. Failed threads read the last word in hash key storage in a loop until it isn't equal to empty key.

