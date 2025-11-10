# Copyright 2024 Yan Yan
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorpc.apps.flow.flowapp import App 


class TemplateCodeApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.class_name_ui = self.root.add_input("ClassName")
        self.is_param_class_ui = self.root.add_switch("IsParamClass")
        self.root.add_buttons(["TVClass", "MemberFunc", "StaticFunc"], self.on_buttons)
        self.root.add_buttons(["TVTensorDefault"], self.on_buttons2)

    async def on_buttons(self, name: str):
        pcls = "pccm.ParameterizedClass" if self.is_param_class_ui.checked else "pccm.Class"
        if name == "TVClass":
            text = f"""
class {self.class_name_ui.value}({pcls}):
    def __init__(self):
        super().__init__()
        self.add_dependency(TensorView)
    
    @pccm.pybind.mark
    @pccm.constructor
    def ctor(self):
        code = pccm.code()
        code.raw(f'''
        ''')
        return code 

    @pccm.destructor
    def dtor(self):
        code = pccm.code()
        code.raw(f'''
        ''')
        return code 
            """
        elif name == "MemberFunc":
            text = f"""    
    @pccm.pybind.mark
    @pccm.member_function
    def function_name(self):
        code = pccm.code()
        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f'''
        ''')
        return code.ret("tv::Tensor")
            """
        elif name == "StaticFunc":
            text = f"""    
    @pccm.pybind.mark
    @pccm.static_function
    def function_name(self):
        code = pccm.code()
        code.arg("a, b, c", "tv::Tensor")
        code.arg("trans_a, trans_b, trans_c", "bool")
        code.arg("arch", "std::tuple<int, int>")
        code.raw(f'''
        ''')
        return code.ret("tv::Tensor")
            """
        else:
            text = ""
        if text != "":
            await self.copy_text_to_clipboard(text)

    async def on_buttons2(self, name: str):
        if name == "TVTensorDefault":
            text = (f"""code.arg("workspace", "tv::Tensor", "tv::Tensor()", "cumm.tensorview.Tensor = Tensor()")""")
        else:
            text = ""
        if text != "":
            await self.copy_text_to_clipboard(text)
