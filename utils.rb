require 'matrix'

class NormalParams
  attr_accessor :avg, :stdev

  def initialize(params)
    @avg = params[:avg]
    @stdev = params[:stdev]
  end
end

class Array
  def avg
    Float(self.reduce(:+))/self.count
  end

  def stdev
    avrg = self.avg
    s_2 = (self.reduce(0.0){|acc, x| acc += Float((x - avrg)**2)})/(self.count - 1)
    return Math.sqrt(s_2)
  end

  def normalize
    avrg = self.avg
    sd = self.stdev
    sd = if sd != 0 then sd else 1 end
    return self.map{|x| (x - avrg)/sd}
  end
end

class Vector
  def transpose
    covector
  end

  def Vector.zeros(size)
    Vector.elements(Array.new(size).fill(0))
  end

  def Vector.ones(size)
    Vector.elements(Array.new(size).fill(1))
  end

  def normalize
    @np = NormalParams.new(:avg=>self.to_a.avg, :stdev=>self.to_a.stdev)
    return Vector.elements(self.to_a.normalize)
  end

  def normal_params
    @np
  end

  def insert(i, val)
    Vector.elements(self.to_a.insert(i, val))
  end
end

class Matrix
  def normalize_column(i)
    cols = column_vectors
    cols[i] = cols[i].normalize
    (@nps ||= [])[i] = cols[i].normal_params
    return Matrix.columns(cols)
  end

  def normalize_columns
    cols = column_vectors
    @nps ||= []
    cols.each_with_index do |col, i|
      cols[i] = col.normalize
      @nps[i] = col.normal_params
    end
    return Matrix.columns(cols)
  end

  def normal_params(i = -1)
    if i>=0 then @nps[i] else @nps end
  end

  def insert_column(i, ary)
    cols = column_vectors
    cols.insert(i, ary)
    return Matrix.columns(cols)
  end
end