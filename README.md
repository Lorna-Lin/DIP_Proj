# DigitalImageProcess多功能图像处理平台

## 简介

2023Spring数字图像处理课程的结课项目，通过基本的图像处理函数实现对图片的尺寸、灰度、锐化的调节以及各类滤镜、特效的添加。

## 技术栈

前端：vue

后端：Django restframework

数据库：Django自带的sqlite

## 配置&使用方法

使用IDE打开项目文件夹，为激活本项目具体使用的环境，需切换到ImageProcess/DjangoRestframework/requirements.txt目录下在终端执行该命令加载项目所需的依赖包。

### 启动后端

迁移数据库：

```python
python manage.py makemigrations
python manage.py migrate
```

切换路径到DjangoRestframework文件夹下：

```python
python manage.py runserver
```

### 启动前端

切换路径到Frontend文件夹下：

```python
npm install
npm run dev
```

最后根据npm运行结果提示，打开localhost即可。

## 项目结构

### 前端

```
src
├─ App.vue
├─ api
│    └─ resolve.js						//所有api接口
├─ components
│    └─ chart							//echarts直方图组件
│           ├─ index.js
│           ├─ index.vue
│           └─ options
├─ main.js
├─ router							
│    └─ index.js
├─ utils
│    └─ request.js						//前端请求封装
└─ views
       └─ home
              ├─ components
              │    ├─ panel.vue			       //操作盘vue
              │    └─ processView.vue	              //处理视图vue
              └─ index.vue
```

### 后端

```
DjangoRestframewor
├─ api
│    ├─ models.py						//数据库表的定义
│    ├─ serializers.py					//处理数据的序列化器
│    ├─ urls.py						//将接口注册到路由中
│    └─ views.py						//各个视图集操作前端传入的数据进行处理
├─ db.sqlite3							//Django自带的数据库
├─ lib
│    ├─ manage
│    │    └─ imageProcess.py			       //所有图像处理函数
│    └─ utils
│         └─ json_response.py			       
├─ log								//运行日志
│    ├─ all.log
│    ├─ error.log
│    └─ script.log
├─ manage.py
├─ media							//本地存放的图片路径
│    └─ images						//处理的图片路径
│           ├─ grass1.jpg
│           ├─ grass1_1475Amh.jpg
│           ├─ grass1_tAABlwO.jpg
│           ├─ ori						//备份的图片路径
│                ├─ grass1.jpg
│                └─ warma4_mt9yOn0.jpg
├─ python_dip_courseproject_django
│    ├─ settings.py 					       //项目各配置
│    └─ urls.py						//项目的总路由
└─ requirements.txt						//项目所需的依赖包
```

## 基本功能

设置了图像直方图供用户参考，设置了复原按钮供用户重置修改

基本图像处理功能分为以下5个模块：

### 尺寸调节

| 函数      | 参数                     | 参数描述            | 功能      |
| --------- | ------------------------ | ------------------- | --------- |
| resize    | zoomXValue、zoomYValue   | x,y轴变化倍率       | 放大/缩小 |
| rotate    | rotateValue              | 角度（-180到180）   | 旋转      |
| reversal  | spinXYVaue               | 翻转，字符串（X/Y） | 翻转      |
| translate | transXValue、transYValue | x,y轴偏移（%）      | 平移      |

### 灰度调节

| 函数          | 参数       | 参数描述      | 功能          |
| ------------- | ---------- | ------------- | ------------- |
| logChange     | None       | None          | 褪色          |
| reverseChange | None       | None          | 反色          |
| gammaChange   | inputGamma | 数值（0到10） | 色彩增强/减淡 |
| contrast      | None       | None          | 对比度加强    |

### 风景滤镜

| 函数       | 参数 | 参数描述 | 功能         |
| ---------- | ---- | -------- | ------------ |
| sepia      | None | None     | 泛黄复古效果 |
| Summer     | None | None     | 夏日温暖效果 |
| Winter     | None | None     | 冬季寒冷效果 |
| hist_equal | None | None     | 霓虹滤镜效果 |

### 磨皮和锐化

| 函数       | 参数                   | 参数描述                    | 功能          |
| ---------- | ---------------------- | --------------------------- | ------------- |
| SmoothSkin | sigmaColor、sigmaSpace | 颜色差值范围                | 磨皮效果      |
| sobel      | ksize                  | Sobel核的大小（1、3、5、7） | sobel锐化效果 |

### 风格特效

| 函数             | 参数                                                     | 参数描述             | 功能         |
| ---------------- | -------------------------------------------------------- | -------------------- | ------------ |
| motion_disk_Blur | inputMotionDistance、inputMotionAngle、inputMotionRadius | 旋转距离、角度、半径 | 动态模糊效果 |
| OilPaint         | None                                                     | None                 | 水彩笔触效果 |
| Manga            | None                                                     | None                 | 细节漫画效果 |
| Mosaic           | None                                                     | None                 | 马赛克效果   |

## 项目概览

![image-20230702194124621](https://lorna-image.oss-cn-shanghai.aliyuncs.com/typora/image-20230702194124621.png)
![image-20230702194048606](https://lorna-image.oss-cn-shanghai.aliyuncs.com/typora/image-20230702194048606.png)
![image-20230702201714769](https://lorna-image.oss-cn-shanghai.aliyuncs.com/typora/image-20230702201714769.png)

## 项目人员

**后端开发、测试：** 林如越 10215101566   **前端设计、测试、报告编写：** 翁佳雯 10215101554
