var scalarField;
var addWarp = true;

/* function to convert a list of numbers (in string form) to a 3d array of numbers */
function listToArray3d(pts) {
    pts = pts.trim().split(" ");

    var array = [];
    var idx = 0;
    for (var i = 0; i < pts.length; i++) {
        if (i % 3 == 0) {
            array[idx] = [];
        }

        array[idx][i % 3] = Number(pts[i]);

        if (i % 3 == 2) {
            idx++;
        }
    }

    return array;
}

/* function to convert a 3d array of numbers to a list of numbers (in string form) */
function array3dToList(arr) {
    var pts = "";

    for (var i = 0; i < arr.length * 3; i++) {
        var x = Math.floor(i / 3);
        var y = i % 3;
        pts += arr[x][y].toString();
        pts += " ";
    }

    return pts;
}

/* function to convert a list of numbers (in string form) to a 1D array of numbers */
function listToArray1d(pts) {
    pts = pts.trim().split(" ");

    var array = [];
    for (var i = 0; i < pts.length; i++) {
        array[i] = Number(pts[i]);
    }

    return array;
}

/* function to convert a 1d array of numbers to a list of numbers (in string form) */
function array1dToList(arr) {
    var pts = "";
    for (var i = 0; i < arr.length; i++) {
        pts += arr[i].toString();
        pts += " ";
    }
}   

/* function to translate colormap and intensity indices into intensity color on the figure */
function addColor() {
    // get information from metadata tag:
    var metadata = $("metadata")[0];
    var colormap = listToArray3d($(metadata).attr("color_map"));
    var indices = listToArray1d($(metadata).attr("indices"));

    // add rgb values to color tag
    var rgb = [];
    for (var i = 0; i < indices.length; i++) {
        var idx = indices[i];
        rgb[i] = colormap[idx];
    }

    // add color node to each indexed face set
    var faces = $("indexedFaceSet");
    for (var i = 0; i < faces.length; i++) {
        // update color node with rgb values
        var color = document.createElement("color");
        $(color).attr("color", array3dToList(rgb));

        var face = faces[i];
        face.appendChild(color);

        // update shape's indexed face set -- remove old face set 
        // (this is because x3dom does not automatically regenerate dom tree based when updated)
        var parent = face.parentElement;
        parent.removeChild(face);
        parent.appendChild(face);
    }
}

/* function to remove color on the figure */
function removeColor() {
    // need to remove the colors from each face set
    var faces = $("indexedFaceSet");
    for (var i = 0; i < faces.length; i++) {
        // remove the color tag
        var color = $(faces[i]).children("color");
        color.remove();

        // re-add the faces to the dom then remove old one (with color) -- x3dom doesn't register updates
        var parent = faces[i].parentElement;
        parent.removeChild(faces[i]);
        parent.appendChild(faces[i]);
    }
}

/* function to calculate the scalar field values at each vertex */
function calculateScalars() {
    // get information from metadata tag:
    var metadata = $("metadata")[0];
    var max_value = Number($(metadata).attr("max_value"));
    var min_value = Number($(metadata).attr("min_value"));
    var indices = listToArray1d($(metadata).attr("indices"));

    // find scale used to normalize color values (there are 256 colors in map by default) 
    var scale = (min_value == max_value) ? 1.0 : 255.0 / (max_value - min_value);

    // calculate scalar value for each index
    var scalars = [];
    for (var i = 0; i < indices.length; i++) {
        scalars[i] = (indices[i] / scale) + min_value;
    }

    return scalars;
}

/* function to warp a shape by a scalar field */
function warpByScalar() {
    if (scalarField.length == 0) {
        return;
    }

    // add warped shapes if not added already
    if (addWarp) {
        var shapes = $("shape");
        var parent = shapes[0].parentElement;

        for (var i = 0; i < shapes.length; i++) {
            // find the points for each shape
            var curr = $(shapes[i]).clone()[0];
            var coord = $(curr).find("coordinate");
            var points = listToArray3d($(coord).attr("point"));

            // change the points according to the scalar field
            for (var j = 0; j < points.length; j++) {
                // FIXME: generalize perpendicular direction
                // change the z-coordinate to be the scalar value
                points[j][2] = scalarField[j];
            }
            $(coord).attr("point", array3dToList(points));

            // add classname to new shape
            $(curr).addClass("warped");

            // add new shape to the parent
            parent.appendChild(curr);
        }

        addWarp = false;
    }

    // adjust the scalar warping by the current scale factor
    var factor = Number($("#warp-slider")[0].value);
    var shapes = $(".warped");
    for (var i = 0; i < shapes.length; i++) {
        var coord = $(shapes[i]).find("coordinate");
        var points = listToArray3d($(coord).attr("point"));

        // change the scalar warping by current scale
        for (var j = 0; j < points.length; j++) {
            points[j][2] = scalarField[j] * factor;
        }
        $(coord).attr("point", array3dToList(points));
    }
}

/* function to determine whether the x3dom shape should be colored */
function shouldColor() {
    var metadata = $("metadata")[0];
    if (!metadata) {
        return false;
    }

    if (metadata.color) {
        return true;
    }

    return false;
}

/* function to call all helper setup functions needed for the menu */
function setupMenu() {
    // first setup the menu buttons
    setupButtons();

    // next setup the content for each subsection
    setupColorContent();
    setupWarpContent();
    // setupOpacityContent();
    setupViewpointContent();
}

/* change the content in the menu based on the button selected */
function setupButtons() {
    $("#menu input[type='radio']").change(function() {
        // get the name from the calling button
        var name = $(this).attr("id");

        // find the content's parent by using the id for the menu content
        var parent = $("#menu-content");

        // find the content from the parent by using the name of calling button
        var selector = "div[for='" + name + "']";
        var content = $(parent).children(selector);
        var children = $(parent).children();

        for (var i = 0; i < children.length; i++) {
            if ($(children[i]).is(content)) {
                $(children[i]).show();
            } else {
                $(children[i]).hide();
            }
        }
    });

    $("#content-options input[type='checkbox']").change(function() {
        // get the name from the calling button
        var name = $(this).attr("id").split("-")[1];
        
        // find the parent for the menu-items
        var parent = $("#menu-items");

        // find the corresponding label element in the menu buttons
        var selector = "label[for='button-" + name +"']"; 
        var label = $(parent).children(selector);

        // hide or show
        if (this.checked) {
            $(label).show();
        } else {
            $(label).hide();
        }
    });
}

/* function to setup the content for the color subsection in menu */
function setupColorContent() {
    // first set up the listener for the 'show color' checkbox
    $("#color-checkbox").change(function() {
        if ($(this).prop("checked")) {
            addColor();
        } else {
            removeColor();
        }
    });

    // build the color map into the menu-content location
    var map_parent = $("#color-map")[0];

    // get information from metadata tag:
    var colormap = listToArray3d($("metadata").attr("color_map"));

    // add spans to the color map parent with the every other color
    for (var i = 0; i < colormap.length; i += 2) {
        var r = Math.round(colormap[i][0] * 256);
        var g = Math.round(colormap[i][1] * 256);
        var b = Math.round(colormap[i][2] * 256);

        var span = document.createElement("span");
        var background_color = "background-color: rgb(" + r + "," + g + "," + b + ")";
        span.style = background_color;
        map_parent.appendChild(span);
    }

    // set minimum and maximum values
    var min_value = Number($("metadata").attr("min_value"));
    $("#min-color-value").html(min_value.toPrecision(3));

    var max_value = Number($("metadata").attr("max_value"));
    $("#max-color-value").html(max_value.toPrecision(3));
}

/* function to setup the content for the warping section in the menu */
function setupWarpContent() {
    // calculate the scalars initially for frame of reference -- global value
    scalarField = calculateScalars();

    // setup the checkbox to toggle the slider
    $("#warp-checkbox").change(function() {
        document.getElementById("warp-slider").disabled = !document.getElementById("warp-slider").disabled;

        if (!document.getElementById("warp-slider").disabled) {
           warpByScalar();
        } else {
            // hide the warped shapes
            $(".warped").remove();
            addWarp = true;
        }
    });

    // setup the slider to change the warped shape and to change the value of the label
    $("#warp-slider").change(function() {
        $("#warp-slider-val").html($(this).val());
        warpByScalar();
    });
}

/* adjust opacity */ 
function adjustOpacity(val) {
    // get the first indexed face set in the scene
    var faces = $("indexedFaceSet")[0];
    var parent = faces.parentElement;
    var material = $(parent).find("material");

    var transparency = 1 - val;
    material.attr("transparency", transparency);
}

function setupOpacityContent() {
    $("#opacity-slider").change(function() {
        $("#opacity-slider-val").html($(this).val());
        adjustOpacity(Number($(this).val()));
    });
}

/* function to setup the content for the viewpoint buttons */
function setupViewpointContent() {
    // add corresponding click listeners for each button
    var buttons = $(".viewpoint");
    for (var i = 0; i < buttons.length; i++) {
        $(buttons[i]).click(function() {
            // the name of the button is the corresponding id of the viewpoint
            var name = $(this).text();
            $("#" + name).attr("set_bind", true);
        });
    }
}

// add color as soon as the document is ready
$(document).ready(function() {
    setupMenu();

    // TODO check if color should be added
    if (shouldColor() || true) {
        addColor();
    }
});