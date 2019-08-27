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
              <svg></svg>
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
import * as d3 from 'd3'
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
    let show = true
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
    drawRelationShip () {
      this.test()
    },
    test () {}
  }
}
var svg = d3.select('svg')
var width = 960
var height = 600

var color = d3.scaleOrdinal(d3.schemeCategory20)

var simulation = d3
  .forceSimulation()
  .force(
    'link',
    d3.forceLink().id(function (d) {
      return d.id
    })
  )
  .force('charge', d3.forceManyBody())
  .force('center', d3.forceCenter(width / 2, height / 2))

d3.json('http://127.0.0.1:5000/api/draw', function (error, graph) {
  if (error) throw error

  var link = svg
    .append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(graph.links)
    .enter()
    .append('line')
    .attr('stroke-width', function (d) {
      return Math.sqrt(d.value)
    })

  var node = svg
    .append('g')
    .attr('class', 'nodes')
    .selectAll('circle')
    .data(graph.nodes)
    .enter()
    .append('circle')
    .attr('r', 5)
    .attr('fill', function (d) {
      return color(d.group)
    })
    .call(
      d3
        .drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
    )

  node.append('title').text(function (d) {
    return d.id
  })

  simulation.nodes(graph.nodes).on('tick', ticked)

  simulation.force('link').links(graph.links)

  function ticked () {
    link
      .attr('x1', function (d) {
        return d.source.x
      })
      .attr('y1', function (d) {
        return d.source.y
      })
      .attr('x2', function (d) {
        return d.target.x
      })
      .attr('y2', function (d) {
        return d.target.y
      })

    node
      .attr('cx', function (d) {
        return d.x
      })
      .attr('cy', function (d) {
        return d.y
      })
  }
})
function dragstarted (d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart()
  d.fx = d.x
  d.fy = d.y
}
function dragged (d) {
  d.fx = d3.event.x
  d.fy = d3.event.y
}
function dragended (d) {
  if (!d3.event.active) simulation.alphaTarget(0)
  d.fx = null
  d.fy = null
}
</script>
