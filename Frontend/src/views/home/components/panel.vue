<template>
  <div class="main">
    <el-tabs v-model="activeName" class="demo-tabs" @tab-click="handleClick">
      <el-tab-pane label="尺寸调节" name="basic" class="el-tabs__content">
        <el-scrollbar noresize height="560px">
          <div style="width: 98%">
            <el-divider content-position="center" style="font-size: 20px">(像素)放大/缩小</el-divider>
            <div class="slider-demo-block">
              <span class="demonstration">X轴变化率(倍)</span>
              <el-slider v-model="zoomXValue" :step="0.1" :min="0.1" :max="10" show-input />
            </div>
            <div class="slider-demo-block">
              <span class="demonstration">Y轴变化率(倍)</span>
              <el-slider v-model="zoomYValue" :step="0.1" :min="0.1" :max="10" show-input />
            </div>
            <el-button @click="resizeHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">平移</el-divider>
            <div class="slider-demo-block">
              <span class="demonstration">X轴偏移(%)</span>
              <el-slider v-model="transXValue" :step="1" :min="-180" :max="180" show-input />
            </div>
            <div class="slider-demo-block">
              <span class="demonstration">Y轴偏移(%)</span>
              <el-slider v-model="transYValue" :step="1" :min="-180" :max="180" show-input />
            </div>
            <el-button @click="translateHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">旋转</el-divider>
            <div class="slider-demo-block">
              <span class="demonstration">角度(%)</span>
              <el-slider v-model="rotateValue" :step="1" :min="-180" :max="180" show-input />
            </div>
            <el-button @click="rotateHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>

            <el-divider content-position="center" style="font-size: 20px">镜像翻转</el-divider>
            <div>
              <el-radio v-model="spinXYVaue" label="X" size="large" border>纵向镜像</el-radio>
              <el-radio v-model="spinXYVaue" label="Y" size="large" border>横向镜像</el-radio>
            </div>
            <br>
            <el-button @click="reversalHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>

          </div>
        </el-scrollbar>
      </el-tab-pane>
      <el-tab-pane label="灰度调节" name="change" class="el-tabs__content">
        <el-scrollbar noresize height="560px">
          <div style="width: 98%">
            <el-divider content-position="center" style="font-size: 20px">褪色效果</el-divider>
<!--            应用对数变换为图片形成褪色效果<br>-->
            <el-button @click="logChangeHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">反色变换</el-divider>
            <el-button @click="reverseChangeHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">对比度加强</el-divider>
            <el-button @click="contrastHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">色彩增强/减弱</el-divider>
            <div class="slider-demo-block">
              <span class="demonstration" style="margin-right: 4px; overflow: visible">参数</span>
              <el-input oninput="if(value>10)value=10;if(value<0)value=0" v-model="inputGamma" placeholder="参数值小于1，色彩减弱；大于1，色彩加强"
                style="margin-left: 10px; width: 600px" />
            </div>

            <el-button @click="gammaChangeHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>

          </div>
        </el-scrollbar>
      </el-tab-pane>
      <el-tab-pane label="风景滤镜" name="noise" class="el-tabs__content">
        <el-scrollbar noresize height="560px">
          <div style="width: 98%">
            <el-divider content-position="center" style="font-size: 20px">泛黄怀旧风</el-divider>
            <el-button @click="sepiaHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">夏日梦幻风</el-divider>
            <el-button @click="summerHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">冬日清冷风</el-divider>
            <el-button @click="winterHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">霓虹幻境风</el-divider>
            <el-button @click="histogramToBalanceHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
          </div>
        </el-scrollbar>
      </el-tab-pane>
      <el-tab-pane label="磨皮和锐化" name="space" class="el-tabs__content">
        <el-scrollbar noresize height="560px">

          <div style="width: 98%">
            <el-divider content-position="center" style="font-size: 20px">磨皮
            </el-divider>
            <div style="margin-bottom: 15px">

            </div>

            <div class="slider-demo-block">
              <span class="demonstration">磨皮强度</span>
              <el-slider v-model="imputsigmaColor" :step="1" :min="1" :max="255" show-input/>
            </div>


            <el-button @click="addSmoothSkinHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting/>
              </el-icon>
              应用
            </el-button>

            <el-divider content-position="center" style="font-size: 20px">锐化
            </el-divider>

            <div class="slider-demo-block">
              <span class="demonstration">锐化强度</span>
              <el-slider v-model="inputSharpenSize" :step="2" :min="1" :max="9" />
            </div>

            <el-button @click="sharpenHandlerTwo" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting/>
              </el-icon>
              应用
            </el-button>

          </div>
        </el-scrollbar>
      </el-tab-pane>
      <el-tab-pane label="风格特效" name="re" class="el-tabs__content">
        <el-scrollbar noresize height="560px">
          <div style="width: 98%">
            <el-divider content-position="center" style="font-size: 20px">动态模糊
            </el-divider>
<!--            <div>-->
<!--              <el-tag style="margin-top: 6px; align-items: center; font-size: 15px">Motion</el-tag>-->
<!--            </div>-->
            <div class="slider-demo-block">
              <span class="demonstration" style="margin-right: 4px; overflow: visible">距离</span>
              <el-slider v-model="inputMotionDistance" :step="1" :min="0" :max="255" show-input />
            </div>
            <div class="slider-demo-block">
              <span class="demonstration" style="margin-right: 4px; overflow: visible">角度</span>
              <el-slider v-model="inputMotionAngle" :step="1" :min="0" :max="360" show-input />
            </div>
<!--            <div>-->
<!--              <el-tag style="margin-top: 6px; align-items: center; font-size: 15px">Disk</el-tag>-->
<!--            </div>-->
            <div class="slider-demo-block">
              <span class="demonstration" style="margin-right: 4px; overflow: visible">半径</span>
              <el-slider v-model="inputMotionRadius" :step="1" :min="0" :max="200" show-input />
            </div>

            <el-button @click="motionHandler" type="primary"
              style="margin-top: 6px; margin-left: 4px; align-items: center">
              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">水彩笔触特效</el-divider>
            <br>
            <el-button @click="addOilPaintHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">

              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">细节漫画特效</el-divider>
            <br>
            <el-button @click="addMangaHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">

              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>
            <el-divider content-position="center" style="font-size: 20px">马赛克风格特效</el-divider>
            <br>
            <el-button @click="addMosaicHandler" type="primary"
                       style="margin-top: 6px; margin-left: 4px; align-items: center">

              <el-icon size="medium">
                <Setting />
              </el-icon>
              应用
            </el-button>

          </div>
        </el-scrollbar>
      </el-tab-pane>

    </el-tabs>
  </div>
</template>

<script>
import * as API from "@/api/resolve";
import { Setting, Calendar } from "@element-plus/icons-vue";
import { ElNotification, ElLoading } from "element-plus";
export default {
  components: {
    Setting,
    Calendar,
  },
  data() {
    return {
      activeName: "basic",

      //放大，缩小
      zoomXValue: 1,
      zoomYValue: 1,

      //旋转
      rotateValue: 0,

      //翻转
      spinXYVaue: "X",

      //平移
      transXValue: 0,
      transYValue: 0,

      //对数变换，无参数
      logChangeValue: "log",

      //反色变换，无参数
      reverseChangeValue: "reverse",

      //幂次变换
      inputGamma: '',

      //直方图均衡化，无参数
      histogramToBalanceValue: 0,

      // //分段线性变换
      // inputA: '',
      // inputB: '',
      // inputC: '',
      // inputD: '',

      //对比度拉伸，无参数
      contrastValue: 0,

      //棕褐滤镜
      inputSepia:'',

      //夏日滤镜
      inputSummer:'',

      //冬日滤镜
      inputWinter:'',

      //磨皮
      imputsigmaColor: '',

      //椒盐噪声
      zoomPepperValue: 0.02,
      zoomSaltValue: 0.02,

      //高斯噪声，均值，方差
      inputMean: '',
      inputVariance: '',

      //Motion距离，角度/Disk半径
      inputMotionDistance: '',
      inputMotionAngle: '',
      inputMotionRadius: '',

      //维纳滤波，平滑约束复原
      inputPSFDistance: '',
      inputPSFAngle: '',
      inputNSRRadius: '',
      ValueOfwienerOrsmooth: 'wiener',

      //自适应中值滤波，无参数（待修改）
      selfMedianValue: 0,

      //自适应均值滤波，无参数（待修改）
      selfMeanValue: 0,

      //Otsu，基于全局阈值
      ValueOfOtsuOrGlobal: "Otsu",

      //区域生长
      ValueOfAreaGrow: "AreaGrow",
      inputAreaGrow: '',

      //边缘检测,阈值
      ValueOfEdge: "Sobel",
      inputEdgeKernel: '',
      inputEdgeThreshold: '',

      //平滑滤波（中值/均值），滤波核大小
      ValueOfMeanOrMedian: "mean",
      inputMeanOrMedianSize: '',

      //锐化滤波，滤波核大小
      ValueOfSharpenOne: "Roberts",
      ValueOfSharpenTwo: "Sobel",
      inputSharpenSize: '',

      //傅里叶变换
      ValueOfmagnitudeOrphase: "magnitude",

      //低通滤波，低通阈值
      ValueOfLowFilter: "ideal",
      inputLowThreshold: '',
      inputLowButter: '',

      //高通滤波，高通阈值
      ValueOfHighFilter: "idealHigh",
      inputHighThreshold: '',
      inputHighButter: '',

      //rgbToHsi
      ValueOfRGBToHSI: "H",

      //彩图分割，阈值
      ValueOfEdgeColor: "Sobel",
      inputEdgeColorKernel: '',
      inputEdgeColorThreshold: '',


    };
  },
  methods: {
    // 测试输入值，将此函数放在下方异步函数中，如addGaussianHandler，通过this.gaussionMean()调用
    gaussianMean() {
      const newVal = this.inputMean
      console.log(newVal);
    },


    async resizeHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.resize({
        zoomXValue: this.zoomXValue,
        zoomYValue: this.zoomYValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "已缩小/放大图片",
        type: "success",
      });
    },

    async rotateHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.rotate({
        rotateValue: this.rotateValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "已旋转图片",
        type: "success",
      });
    },



    async reversalHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.reversal({
        spinXYVaue: this.spinXYVaue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "已翻转图片",
        type: "success",
      });
    },

    async translateHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.translate({
        transXValue: this.transXValue,
        transYValue: this.transYValue,
        id: _id,
      });


      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "已平移图片",
        type: "success",
      });
    },

    async logChangeHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.logChange({
        logChangeValue: this.logChangeValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已褪色处理",
        type: "success",
      });
    },

    async reverseChangeHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.reverseChange({
        reverseChangeValue: this.reverseChangeValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已反转变换",
        type: "success",
      });
    },

    async gammaChangeHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.gammaChange({
        inputGamma: this.inputGamma,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已色彩增强/减弱",
        type: "success",
      });
    },

    async histogramToBalanceHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.histogramToBalance({
        histogramToBalanceValue: this.histogramToBalanceValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入霓虹滤镜",
        type: "success",
      });
    },

    async contrastHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.contrast({
        contrastValue: this.contrastValue,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已增强对比度",
        type: "success",
      });
    },

    async sepiaHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.addSepia({
        inputSepia: this.inputSepia,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入怀旧滤镜",
        type: "success",
      });
    },

    async summerHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.addSummer({
        inputSummer: this.inputSummer,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入夏日滤镜",
        type: "success",
      });
    },

    async winterHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.addWinter({
        inputWinter: this.inputWinter,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入冬日滤镜",
        type: "success",
      });
    },



    async motionHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.motion({
        inputMotionDistance: this.inputMotionDistance,
        inputMotionAngle: this.inputMotionAngle,
        inputMotionRadius: this.inputMotionRadius,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已动态模糊",
        type: "success",
      });
    },

    //油画效果
    async addOilPaintHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作
      //针对不同操作调用不同API即可
      let res = await API.addOilPaint({

        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入水彩特效",
        type: "success",
      });
    },

    //漫画效果
    async addMangaHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作
      //针对不同操作调用不同API即可
      let res = await API.addManga({

        id: _id,
      });
      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入漫画特效",
        type: "success",
      });
    },

      //马赛克效果
      async addMosaicHandler() {
        let loading = ElLoading.service({
          lock: true,
          text: "处理中...",
          background: "rgba(255, 255, 255, 0.2)",
        });
        let _id = this.$store.getters.id;
        //以上为必备操作
        //针对不同操作调用不同API即可
        let res = await API.addMosaic({

          id: _id,
        });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已加入马赛克特效",
        type: "success",
      });
    },


    //磨皮双边滤波
    async addSmoothSkinHandler() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作
      //针对不同操作调用不同API即可
      let res = await API.addSmoothSkin({
        imputsigmaColor: this.imputsigmaColor,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已磨皮处理",
        type: "success",
      });
    },



    async sharpenHandlerOne() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.sharpenOne({
        ValueOfSharpenOne: this.ValueOfSharpenOne,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已经过锐化处理",
        type: "success",
      });
    },

    async sharpenHandlerTwo() {
      let loading = ElLoading.service({
        lock: true,
        text: "处理中...",
        background: "rgba(255, 255, 255, 0.2)",
      });
      let _id = this.$store.getters.id;
      //以上为必备操作

      //针对不同操作调用不同API即可
      let res = await API.sharpenTwo({
        ValueOfSharpenTwo: this.ValueOfSharpenTwo,
        inputSharpenSize: this.inputSharpenSize,
        id: _id,
      });

      //以下为必备操作
      this.$store.commit("image/SET_URL", res.data.file);
      this.$forceUpdate();
      this.$emit("refresh");
      loading.close();
      ElNotification({
        title: "操作成功",
        message: "图片已经过锐化处理",
        type: "success",
      });
    },







  },
};
</script>

<style scoped>
.slider-demo-block {
  display: flex;
  align-items: center;
}

.slider-demo-block .el-slider {
  margin-top: 0;
  margin-left: 12px;
}

.slider-demo-block .demonstration {
  font-size: 14px;
  line-height: 44px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 0;
}

.slider-demo-block .demonstration+.el-slider {
  flex: 1 0 60%;
}

.el-tabs__content {
  padding: 4px;
  color: #6b778c;
}

.main {
  width: 590px;
  height: 650px;
}
</style>
