<template>
  <div id="components-layout-demo-basic">
    <a-layout>
      <a-layout-header theme='light' style="color: white; font-size: 20px">
        新 闻 人 物 言 论 自 动 提 取
      </a-layout-header>
      <div style='background: #fff;'><br><br><br></div>
      <a-layout>
        <a-layout-sider theme='light'></a-layout-sider>
        <a-layout-content theme='light'>
          <!-- <button type="button" @click="getRandomFromBackend">get new random number</button>
          <p>{{ randomNumber }}</p> -->
          <a-input-search placeholder="输入句子……" @search="parse" enterButton="提取" size="large" />
          <div style='background: #fff; padding-top: 20px'>
            <a-tag closable @close="log" color='green' v-if="whoShow">{{ who }}</a-tag>
            <a-tag closable @close="log" color='purple' v-if="speechShow">{{ speech }}</a-tag>
          </div>
        </a-layout-content>
        <a-layout-sider theme='light'></a-layout-sider>
      </a-layout>
      <!-- <a-layout-footer theme='light'>Footer</a-layout-footer> -->
    </a-layout>
  </div>
</template>

<script>
import axios from 'axios'
export default {
  data () {
    let randomNumber = 39
    let whoShow = false
    let speechShow = false
    let who = ''
    let speech = ''
    // let speech = {person: [], speeches: []}
    return {
      randomNumber,
      who,
      speech,
      whoShow,
      speechShow
    }
  },
  methods: {
    getRandomFromBackend () {
      const path = `http://localhost:5000/api/random`
      axios.get(path)
        .then(response => {
          this.randomNumber = response.data.randomNumber
        })
        .catch(error => {
          console.log(error)
        })
    },
    parse (sentence) {
      const path = `http://localhost:5000/api/extract`
      axios.get(path, {
        params: {
          sentence
        }
      })
        .then(response => {
          let data = response.data
          if (data.who !== null) {
            this.whoShow = true
            this.speechShow = true
          }
          this.who = data.who
          this.speech = data.speech
          console.log(data.who)
        })
        .catch(error => {
          console.log(error)
        })
    }
  }
}
</script>
