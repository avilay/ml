require 'test/unit'
require 'matrix'
require_relative '../lin_reg/linear_regression'

class LinearRegressionTest < Test::Unit::TestCase

  def test_all_zeros
    x_mat = Matrix[[0,0], [0,0], [0,0], [0,0], [0,0]]
    y_vec = Vector[0, 0, 0, 0, 0]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_equal(0.0, lin_reg.cost)
  end

  def test_all_neg
    x_mat = Matrix[[-6,-4], [-9,-4], [-6,0], [-7,-2], [-3,-3]]
    y_vec = Vector[-77, -113, -69, -85, -39]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_in_delta(0.007, lin_reg.cost, 0.1)
  end

  def test_all_pos
    x_mat = Matrix[[5,17], [3,12], [1,18], [0,18], [0,12]]
    y_vec = Vector[97, 63, 51, 39, 27]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.03)
    assert_in_delta(0.018, lin_reg.cost, 0.1)
  end

  def test_mix
    x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
    y_vec = Vector[7, 99, -37, 129, -99]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(1)
    assert_in_delta(0.0008, lin_reg.cost, 0.1)    
  end

  def test_same_dist
    x_mat = Matrix[[2,800], [3,900], [4,1000], [5,1100], [6,1200]]
    y_vec = Vector[290, 340, 390, 440, 490]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    lin_reg.train(0.1)
    assert_in_delta(0.002, lin_reg.cost, 0.1)    
  end

  def test_large_alpha
    x_mat = Matrix[[-1,8], [6,12], [-6,16], [8,15], [-8,-3]]
    y_vec = Vector[7, 99, -37, 129, -99]
    lin_reg = LinearRegression.new(x_mat, y_vec)
    lin_reg.prepare_data
    assert_raises(ArgumentError){lin_reg.train(30)}
  end
end