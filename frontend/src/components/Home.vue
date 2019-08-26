<template>
  <div id="components-layout-demo-basic">
    <a-layout>
      <a-layout-header theme="light" style="color: white; font-size: 20px">新 闻 人 物 言 论 自 动 提 取</a-layout-header>
      <div style="background: #fff">
        <br />
        <br />
        <br />
      </div>
      <a-layout>
        <a-layout-sider theme="light"></a-layout-sider>
        <a-layout-content theme="light">
          <a-textarea
            placeholder="请输入待提取新闻……"
            :autosize="{ minRows: 2, maxRows: 6 }"
            v-model="sentence"
          />
          <div style="background: #fff">
            <br />
          </div>
          <a-button type="primary" @click="extract" block>提取</a-button>
          <div style="background: #fff">
            <br />
            <br />
          </div>
          <a-row style="background: #fff" v-if="show">
            <a-col :span="12">
              <a-table
                :columns="columns"
                :dataSource="data"
                :pagination="{ pageSize: 3 }"
                :scroll="{ y: 240 }"
                style="background: #fff"
              />
            </a-col>
            <a-col :span="12">
              <div id="visualization"></div>
            </a-col>
          </a-row>
        </a-layout-content>
        <a-layout-sider theme="light"></a-layout-sider>
      </a-layout>
    </a-layout>
  </div>
</template>

<script>
import axios from 'axios'
const columns = [
  {
    title: '人物',
    dataIndex: 'person',
    width: 50
  },
  {
    title: '谓词',
    dataIndex: 'predicate',
    width: 50
  },
  {
    title: '言论',
    dataIndex: 'speech',
    width: 150
  }
]
export default {
  data () {
    let data = []
    let sentence = ''
    let show = false
    return {
      columns,
      data,
      sentence,
      show
    }
  },
  watch: {
    sentence: function () {
      if (this.sentence.length === 0) {
        this.show = false
      }
    }
  },
  methods: {
    extract (e) {
      this.show = true
      this.parse(this.sentence)
      this.drawRelationShip()
    },
    parse (sentence) {
      const path = `http://localhost:5000/api/extract`
      axios
        .get(path, {
          params: {
            sentence
          }
        })
        .then(response => {
          let resp = response.data
          let predicates = resp.predicates
          let persons = resp.persons
          let speeches = resp.speeches
          let tmp = []
          for (let i = 0; i < persons.length; i++) {
            tmp.push({
              key: i,
              person: persons[i],
              predicate: predicates[i],
              speech: speeches[i]
            })
          }
          this.data = tmp
        })
        .catch(error => {
          console.log(error)
        })
    },
    drawRelationShip () {}
  }
}
</script>
