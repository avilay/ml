require 'test/unit'
require 'matrix'
require_relative '../lin_reg/linear_regression'

class LinearRegressionTest < Test::Unit::TestCase

  def test_train_all_zeros
    x_mat = Matrix[[0,0], [0,0], [0,0], [0,0], [0,0]]
    y_vec = Vector[0, 0, 0, 0, 0]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_equal(0.0, lin_reg.cost)
  end

  def test_train_all_neg
    x_mat = Matrix[[-6,-4], [-9,-4], [-6,0], [-7,-2], [-3,-3]]
    y_vec = Vector[-77, -113, -69, -85, -39]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_in_delta(0.007, lin_reg.cost, 0.1)
  end

  def test_train_all_pos
    x_mat = Matrix[[5,17], [3,12], [1,18], [0,18], [0,12]]
    y_vec = Vector[97, 63, 51, 39, 27]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.03)
    assert_in_delta(0.018, lin_reg.cost, 0.1)
  end

  def test_train_mix
    x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
    y_vec = Vector[7, 99, -37, 129, -99]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(1)
    assert_in_delta(0.0008, lin_reg.cost, 0.1)    
  end

  def test_train_same_dist
    x_mat = Matrix[[2,800], [3,900], [4,1000], [5,1100], [6,1200]]
    y_vec = Vector[290, 340, 390, 440, 490]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_in_delta(0.002, lin_reg.cost, 0.1)    
  end

  def test_train_large_alpha
    x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
    y_vec = Vector[7, 99, -37, 129, -99]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    assert_raises(ArgumentError){lin_reg.train(30)}
  end

  def train_model
    x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
    y_vec = Vector[11, 89, -23, 115, -81]
    @lin_reg = LinearRegression.new(x_mat, y_vec)
    @lin_reg.prepare_data
    @lin_reg.train(1)
  end

  def single_instance_asserts
    assert_in_delta(11, @lin_reg.predict([-1,8]), 0.11)
    assert_in_delta(89, @lin_reg.predict(Vector[6,12]), 0.11)
    assert_in_delta(-23, @lin_reg.predict([-6,16]), 0.11)
    assert_in_delta(115, @lin_reg.predict([8,15]), 0.11)
    assert_in_delta(-81, @lin_reg.predict(Vector[-8,-3]), 0.11)
    assert_in_delta(3005, @lin_reg.predict([100,1000]), 10)
  end

  def setup_model
    theta_vec = Vector[22.2, 70.79118516220045, 15.4620715673726]
    normal_params = []
    normal_params[1] = NormalParams.new(avg: -0.2, stdev: 7.08519583356734)
    normal_params[2] = NormalParams.new(avg: 9.6, stdev: 7.700649323271382)
    return LinearRegressionModel.new(theta_vec: theta_vec, normal_params: normal_params)
  end

  def test_predict_single_instance
    train_model
    single_instance_asserts
  end

  def test_predict_with_model
    @lin_reg = LinearRegression.new
    @lin_reg.model = setup_model
    single_instance_asserts
  end

  def multiple_instance_asserts
    x_mat = Matrix[[-5,10], [-7,32], [0,4]]
    actual = @lin_reg.predict(x_mat)
    expected = [-25, -1, 13]
    actual.to_a.each_with_index do |y, i|
      assert_in_delta(expected[i], y, 0.5)
    end
  end

  def test_predict_multiple_instances
    train_model
    multiple_instance_asserts
  end

  def test_predict_multiple_instances_with_model
    @lin_reg = LinearRegression.new
    @lin_reg.model = setup_model
    multiple_instance_asserts
  end

  def test_debug_print
    train_model
    @lin_reg.debug_print
    assert(true) #As long as debug_print doesnt throw we are good
  end

  def test_analytics
    train_model
    assert_equal(14, @lin_reg.analytics.num_iters)
    assert_equal(1, @lin_reg.analytics.alpha)
    assert_in_delta(253, @lin_reg.analytics.cost_history[0], 0.5)
    assert_in_delta(0.0005, @lin_reg.analytics.cost_history[-1], 0.001)
  end
end