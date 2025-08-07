import json, uuid
from IPython.display import display, HTML


def interactive_plot_with_options(figure):
    """
    Display an interactive Plotly figure with sliders for atom size and opacity.

    Parameters:
    - figure: A Plotly go.Figure object

    returns:
    - None
    """

    fig_json = figure.to_json()

    # Generate unique IDs for HTML elements
    # This avoids conflicts when multiple plots are displayed in the same notebook
    # or when the function is called multiple times.
    div_id = f"plotly-div-{uuid.uuid4().hex}"
    size_slider_id = f"size-slider-{uuid.uuid4().hex}"
    size_val_id = f"size-val-{uuid.uuid4().hex}"
    opacity_slider_id = f"size-slider-{uuid.uuid4().hex}"
    opacity_val_id = f"opacity-val-{uuid.uuid4().hex}"

    html = f"""
<div>
  <!-- Sliders for atom size and opacity -->
  <label>Atom Size: <span id="{size_val_id}">10</span></label>
  <input id="{size_slider_id}" type="range" min="1" max="30" value="10" step="1" style="width:300px;">
  <br>
  <label>Opacity: <span id="{opacity_val_id}">1.00</span></label>
  <input id="{opacity_slider_id}" type="range" min="0" max="1" value="1" step="0.01" style="width:300px;">
</div>
<div id="{div_id}" style="width:800px;height:600px;"></div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
  var fig = {fig_json};
  Plotly.newPlot('{div_id}', fig.data, fig.layout).then(function(gd) {{
    // 1) Initial camera annotation
    Plotly.relayout(gd, {{
      'annotations[0]': {{
        x: 0, y: 1, xref: 'paper', yref: 'paper',
        text: 'Camera: …', showarrow: false,
        bgcolor: 'rgba(255,255,255,0.7)', bordercolor: 'black',
        font: {json.dumps({"size":12})}
      }}
    }});

    // 2) Sliders elements
    var sizeSlider = document.getElementById('{size_slider_id}');
    var sizeVal    = document.getElementById("{size_val_id}");
    var opacSlider = document.getElementById('{opacity_slider_id}');
    var opacVal    = document.getElementById("{opacity_val_id}");

    // 3) Update marker sizes & opacity
    function updateMarkers() {{
      var newSize = +sizeSlider.value;
      var newOpac = +opacSlider.value;
      sizeVal.innerText = newSize;
      opacVal.innerText = newOpac.toFixed(2);

      var update = {{ 'marker.size': [], 'marker.opacity': [] }};
      fig.data.forEach(function(trace) {{
        if (trace.mode && trace.mode.includes('markers')) {{
          update['marker.size'].push(newSize);
          update['marker.opacity'].push(newOpac);
        }} else {{
          update['marker.size'].push(null);
          update['marker.opacity'].push(null);
        }}
      }});
      Plotly.restyle(gd, update);
    }}

    sizeSlider.addEventListener('input', updateMarkers);
    opacSlider.addEventListener('input', updateMarkers);

    // 4) Update camera annotation on camera move
    gd.on('plotly_relayout', function(eventdata) {{
      if (eventdata['scene.camera']) {{
        var cam = gd.layout.scene.camera;
        var txt = 'eye:    (' + cam.eye.x.toFixed(2) + ', ' + cam.eye.y.toFixed(2) + ', ' + cam.eye.z.toFixed(2) + ')<br>'
                + 'center: (' + cam.center.x.toFixed(2) + ', ' + cam.center.y.toFixed(2) + ', ' + cam.center.z.toFixed(2) + ')<br>'
                + 'up:     (' + cam.up.x.toFixed(2) + ', ' + cam.up.y.toFixed(2) + ', ' + cam.up.z.toFixed(2) + ')';
        Plotly.relayout(gd, {{ 'annotations[0].text': txt }});
      }}
    }});
  }});
</script>
    """
    display(HTML(html))