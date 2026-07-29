(() => {
    "use strict";
    const root = document.getElementById("annotation-app");
    if (!root || typeof Konva === "undefined") return;
    const classSelector = document.getElementById("class-selector");
    const status = document.getElementById("annotation-status");
    const boxes = [];
    let stage;
    let layer;
    let transformer;
    let drawing = true;
    let draft = null;
    let start = null;
    let displayScale = 1;
    let naturalWidth = 0;
    let naturalHeight = 0;

    const colorFor = (name) => {
        let hash = 0;
        for (const character of name) hash = ((hash << 5) - hash) + character.charCodeAt(0);
        return `hsl(${Math.abs(hash) % 360}, 80%, 50%)`;
    };
    const syncLabel = (entry) => entry.label.position({x: entry.rect.x(), y: Math.max(0, entry.rect.y() - 20)});
    const select = (rect) => {
        transformer.nodes(rect ? [rect] : []);
        layer.draw();
    };
    const addBox = (data) => {
        const color = colorFor(data.class_name);
        const rect = new Konva.Rect({
            x: data.x_min * displayScale,
            y: data.y_min * displayScale,
            width: (data.x_max - data.x_min) * displayScale,
            height: (data.y_max - data.y_min) * displayScale,
            stroke: color, strokeWidth: 2, draggable: true,
            className: data.class_name, source: data.source || "human", confidence: data.confidence,
        });
        const confidence = data.confidence == null ? "" : ` ${(data.confidence * 100).toFixed(1)}%`;
        const label = new Konva.Text({text: `${data.class_name}${confidence}`, fill: "white", fontSize: 14, padding: 3, x: rect.x(), y: Math.max(0, rect.y() - 20)});
        label.fill(color);
        const entry = {rect, label};
        boxes.push(entry);
        layer.add(rect, label);
        rect.on("click tap", (event) => { event.cancelBubble = true; select(rect); });
        rect.on("dragmove", () => syncLabel(entry));
        rect.on("transformend", () => {
            rect.width(Math.max(2, rect.width() * rect.scaleX()));
            rect.height(Math.max(2, rect.height() * rect.scaleY()));
            rect.scale({x: 1, y: 1});
            syncLabel(entry);
        });
        return rect;
    };
    const deleteSelected = () => {
        const selected = transformer.nodes()[0];
        if (!selected) return;
        const index = boxes.findIndex((entry) => entry.rect === selected);
        if (index >= 0) {
            boxes[index].rect.destroy();
            boxes[index].label.destroy();
            boxes.splice(index, 1);
        }
        select(null);
    };
    const save = async () => {
        status.textContent = "Saving…";
        const annotations = boxes.map(({rect}) => ({
            class_name: rect.getAttr("className"),
            x_min: rect.x() / displayScale,
            y_min: rect.y() / displayScale,
            x_max: (rect.x() + rect.width()) / displayScale,
            y_max: (rect.y() + rect.height()) / displayScale,
            source: rect.getAttr("source") || "human",
            confidence: rect.getAttr("confidence"),
        }));
        const response = await fetch(root.dataset.saveUrl, {
            method: "PUT", headers: {"Content-Type": "application/json", Accept: "application/json"},
            body: JSON.stringify({image: root.dataset.imageName, image_width: naturalWidth, image_height: naturalHeight, annotations}),
        });
        const result = await response.json();
        status.textContent = response.ok ? `Saved ${result.saved} box(es)` : result.error;
    };
    const load = async () => {
        const [image, response] = await Promise.all([
            new Promise((resolve, reject) => { const item = new Image(); item.onload = () => resolve(item); item.onerror = reject; item.src = root.dataset.imageUrl; }),
            fetch(root.dataset.apiUrl, {headers: {Accept: "application/json"}}),
        ]);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        naturalWidth = image.naturalWidth;
        naturalHeight = image.naturalHeight;
        displayScale = Math.min(1, 1000 / naturalWidth, 650 / naturalHeight);
        stage = new Konva.Stage({container: root, width: naturalWidth * displayScale, height: naturalHeight * displayScale});
        layer = new Konva.Layer();
        transformer = new Konva.Transformer({rotateEnabled: false, keepRatio: false, boundBoxFunc: (oldBox, newBox) => newBox.width < 3 || newBox.height < 3 ? oldBox : newBox});
        layer.add(new Konva.Image({image, width: stage.width(), height: stage.height(), listening: false}));
        stage.add(layer);
        data.annotations.forEach(addBox);
        layer.add(transformer);
        stage.on("click tap", (event) => { if (event.target === stage) select(null); });
        stage.on("mousedown touchstart", (event) => {
            if (!drawing || event.target !== stage) return;
            start = stage.getPointerPosition();
            draft = new Konva.Rect({x: start.x, y: start.y, width: 0, height: 0, stroke: colorFor(classSelector.value), strokeWidth: 2, dash: [6, 4]});
            layer.add(draft);
        });
        stage.on("mousemove touchmove", () => {
            if (!draft || !start) return;
            const pointer = stage.getPointerPosition();
            draft.position({x: Math.min(start.x, pointer.x), y: Math.min(start.y, pointer.y)});
            draft.size({width: Math.abs(pointer.x - start.x), height: Math.abs(pointer.y - start.y)});
        });
        stage.on("mouseup touchend", () => {
            if (!draft) return;
            const shape = {x: draft.x(), y: draft.y(), width: draft.width(), height: draft.height()};
            draft.destroy(); draft = null; start = null;
            if (shape.width >= 3 && shape.height >= 3) select(addBox({class_name: classSelector.value, x_min: shape.x / displayScale, y_min: shape.y / displayScale, x_max: (shape.x + shape.width) / displayScale, y_max: (shape.y + shape.height) / displayScale, source: "human", confidence: null}));
        });
        status.textContent = `${boxes.length} saved box(es) loaded`;
        root.focus();
    };
    document.getElementById("draw-tool").addEventListener("click", () => { drawing = true; status.textContent = "Draw mode active"; });
    document.getElementById("delete-box").addEventListener("click", deleteSelected);
    document.getElementById("save-annotations").addEventListener("click", () => save().catch((error) => { status.textContent = error.message; }));
    document.addEventListener("keydown", (event) => {
        if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
        if (event.key.toLowerCase() === "b") drawing = true;
        else if (event.key === "Delete") deleteSelected();
        else if (event.key === "Enter") save().catch((error) => { status.textContent = error.message; });
        else if (event.key.toLowerCase() === "n") window.location.href = document.getElementById("next-image").href;
        else if (event.key.toLowerCase() === "p") window.location.href = document.getElementById("previous-image").href;
        else if (/^[1-9]$/.test(event.key) && classSelector.options[event.key - 1]) classSelector.selectedIndex = Number(event.key) - 1;
    });
    load().catch((error) => { status.textContent = `Unable to load editor: ${error.message}`; });
})();
