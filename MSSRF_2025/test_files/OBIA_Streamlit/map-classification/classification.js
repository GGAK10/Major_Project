const map = L.map('map').setView([37.7749, -122.4194], 10);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

async function loadModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
}

async function trainModel(model) {
    const xs = tf.tensor2d([[37.7749, -122.4194], [37.8044, -122.2711], [37.6879, -122.4702]]);
    const ys = tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    await model.fit(xs, ys, { epochs: 50 });
}

(async () => {
    const model = await loadModel();
    await trainModel(model);
})();

