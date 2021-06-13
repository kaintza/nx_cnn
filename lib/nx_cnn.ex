defmodule NxCnn do
  require Axon

  @size {224, 224}
  @batch_size 256

  def save(model, path \\ "data.bin") do
    bytes = :erlang.term_to_binary(model)
    File.write!(path, bytes)
  end

  def load(path \\ "data.bin") do
    File.read!(path)
    |> :erlang.binary_to_term
  end

  def train_it(path, count, epochs) do
    faces = NxCnn.read_data_from_xlsx(path, count)

    IO.inspect(faces |> length)

    genders = faces |> Enum.map(fn %{gender: gender} -> gender end)
    paths = faces |> Enum.map(fn %{path: path} -> path end)

    train_images = NxCnn.convert_images_to_tensor(paths)
    train_labels = NxCnn.convert_labels_to_tensor(genders)

    {x, y} = @size

    train_images
    |> hd()
    |> Nx.slice_axis(0, 1, 0)
    |> Nx.reshape({3, x, y})
    |> Nx.mean(axes: [0], keep_axes: true)
    |> Nx.to_heatmap()
    |> IO.inspect

    IO.inspect model

    NxCnn.train(train_images, train_labels, epochs)
  end

  def train(train_images, train_labels, epochs \\ 20) do
    state =
      model
      |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.sgd(0.01), metrics: [:accuracy])
      |> Axon.Training.train(train_images, train_labels, epochs: epochs, compiler: EXLA)

    state |> save

    state
  end

  def predict(final_training_state, test_images) do
    model
    |> Axon.predict(final_training_state[:params], test_images)
    # |> Nx.argmax(axis: -1)
  end

  def simple_model do
    {x, y} = @size

    Axon.input({nil, 3, x, y})
    |> Axon.conv(16, kernel_size: {7, 7}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.conv(16, kernel_size: {7, 7}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.activation(:relu)
    |> Axon.avg_pool(kernel_size: {2, 2}, padding: :same, strides: 2)
    |> Axon.dropout()
    |> Axon.conv(32, kernel_size: {5, 5}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.conv(32, kernel_size: {5, 5}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.activation(:relu)
    |> Axon.avg_pool(kernel_size: {2, 2}, padding: :same, strides: 2)
    |> Axon.dropout()
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.activation(:relu)
    |> Axon.avg_pool(kernel_size: {2, 2}, padding: :same, strides: 2)
    |> Axon.dropout()
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu, padding: :same)
    |> Axon.batch_norm()
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.activation(:relu)
    |> Axon.avg_pool(kernel_size: {2, 2}, padding: :same, strides: 2)
    |> Axon.dropout()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.activation(:relu)
    |> Axon.avg_pool(kernel_size: {2, 2}, padding: :same, stride: 2)
    |> Axon.dropout()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu, padding: :same)
    # |> Axon.batch_norm()
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(2, activation: :softmax)
  end

  def vgg_model do
    {x, y} = @size

    Axon.input({nil, 3, x, y})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2)
    |> Axon.batch_norm()
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(256, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(512, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(512, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2)
    |> Axon.conv(512, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.conv(512, kernel_size: {3, 3}, activation: :relu)
    |> Axon.conv(4096, kernel_size: {7, 7}, activation: :relu)
    |> Axon.dropout()
    |> Axon.conv(4096, kernel_size: {1, 1}, activation: :relu)
    |> Axon.dropout()
    |> Axon.conv(2622, kernel_size: {7, 7}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(2, activation: :softmax)
  end

  def model do
    {x, y} = @size

    Axon.input({nil, 3, x, y})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    # |> Axon.batch_norm()
    |> Axon.spatial_dropout()
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2)
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    # |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2)
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    # |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2}, strides: 2)
    |> Axon.conv(128, kernel_size: {3, 3}, activation: :relu)
    # |> Axon.batch_norm()
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(2, activation: :softmax)
  end

  def read_data_from_xlsx(path, rows \\ 50000) do
    {:ok, r} = Xlsxir.peek(path, 0, rows + trunc(rows * 0.25))
    [hd | face_list] = Xlsxir.get_list(r)

    face_list
    |> Enum.reject(fn [_id, _dob, gender, face_score, _second_face_score, _image_path] ->
      gender == "NaN" || face_score == "#NÉV?"
    end)
    |> Enum.map(fn [_id, dob, gender, _face_score, _second_face_score, image_path] ->
      %{gender: gender, path: image_path, dob: dob}
    end)
    |> Enum.take_random(rows)
  end

  def image_bin(url) do
    %{body: body} = Scidata.Utils.get!(url)
    body
  end

  def convert_images_to_tensor(image_paths) do
    base_path = "/home/adamk/Downloads/imdb_crop/"

    pixels =
      image_paths
      |> Enum.map(fn p ->
        convert_image_from_path(base_path <> p)
      end)

    # |> Enum.map(fn path -> Task.async(fn -> convert_image_from_path(base_path <> path) end) end)
    # |> Enum.map(&Task.await(&1, :infinity))

    {x, y} = @size

    pixels
    |> :binary.list_to_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({length(pixels), 3, x, y})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(@batch_size)
  end

  def convert_labels_to_tensor(labels) do
    labels
    |> Nx.tensor
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..1)))
    |> Nx.to_batched_list(@batch_size)
  end

  def convert_image_from_url(url, cache \\ true) do
    %{body: body} = Scidata.Utils.get!(url)
    tmp_path = "/tmp/image.png"

    File.write!(tmp_path, body)

    tmp_path |> convert_image_from_path(cache)
  end

  def test_image_from_url(url) do
    {x, y} = @size

    url
    |> convert_image_from_url(false)
    |> :binary.list_to_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({1, 3, x, y})
    |> Nx.divide(255.0)
  end

  def test_image_from_path(path) do
    {x, y} = @size

    path
    |> convert_image_from_path
    |> :binary.list_to_bin
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({1, 3, x, y})
    |> Nx.divide(255.0)
  end

  def convert_image_from_path(path, cache \\ true) do
    filename = path |> String.trim(".jpg") |> String.split("/") |> List.last
    {x, y} = @size
    converted_tmp = "/tmp/#{filename}_resized_#{x}x#{y}.png"

    # converted_tmp = "/tmp/output.png"

    size_parameter = "#{x}x#{y}!"

    if !File.exists?(converted_tmp) || !cache do
      IO.puts("nincs")
      System.cmd("convert", [path, "-resize", size_parameter, converted_tmp])
    end

    image =
      converted_tmp
      |> Mogrify.open
      |> Mogrify.format("png")
      |> Mogrify.save

    {:ok, %{data: image_bin}} = Pixels.read_file(image.path)

    pixels = image_bin |> :binary.bin_to_list |> Enum.chunk_every(4)

    r = pixels |> Enum.map(fn [r, g, b, a] -> r end)
    g = pixels |> Enum.map(fn [r, g, b, a] -> g end)
    b = pixels |> Enum.map(fn [r, g, b, a] -> b end)

    r ++ g ++ b
  end
end
