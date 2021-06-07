function createChart (data) {
  var data = data;
  // data = JSON.parse("["+data.toString()+"]")
  console.log(data)
  var chart = JSC.chart('chartDiv', { 
      debug: true, 
      type: 'line', 
      title_label_text: 'Line Series Types', 
      legend_position: 'inside bottom right', 
      toolbar_items: { 
        'Line Type': { 
          type: 'select', 
          label_style_fontSize: 13, 
          margin: 5, 
          items: 'Line,Step,Spline', 
          events_change: function(val) { 
            chart.series().options({ type: val }); 
          } 
        } 
      }, 
      xAxis: { scale_type: 'time' }, 
      series: data 
    });
} 