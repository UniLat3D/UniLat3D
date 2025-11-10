isort --atomic . && \
yapf -i --recursive -vv ./cumm ./test
yapf -i -vv setup.py
find ./include -regex '.*\.\(cpp\|hpp\|cc\|cxx\|cu\|cuh\|h\)' | xargs clang-format -i