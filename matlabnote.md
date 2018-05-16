## matlab

## simulink

### 模块方法
gcb 获取当前模块
gcbh 获取当前模块句柄
get()获取模块所有属性，参数是句柄和属性名称
set()设置属性
get_param()获取指定属性，第一个参数是模块路径
gcs returns the full pathname of the current system.
### 模块属性-Position

get(gcbh,'Position')
或
get_param('untitled/Integrator','Position')

vector of coordinates, in pixels: [left top right bottom]四个边界确定模块的位置