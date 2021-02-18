#pragma once

template< typename real, int size >
class vector
{
public:
	vector() = default;
	vector(const vector&) = default;
	vector(vector&&) = default;

	template< typename scalar >
	vector(scalar c)
	{
		for (int i = 0; i < size; i++)
			data[i] = c;
	}

	static constexpr int getSize()
	{
		return size;
	}

	vector& operator=(const vector&) = default;
	vector& operator=(vector&&) = default;

	real& operator[](int i)
	{
		return data[i];
	}

	const real& operator[](int i) const
	{
		return data[i];
	}

	vector& operator+=(const vector& v)
	{
		for (int i = 0; i < size; i++)
			data[i] += v[i];
		return *this;
	}

	vector& operator-=(const vector& v)
	{
		for (int i = 0; i < size; i++)
			data[i] += v[i];
		return *this;
	}

	vector& operator*=(const vector& v)
	{
		for (int i = 0; i < size; i++)
			data[i] *= v[i];
		return *this;
	}

	vector& operator/=(const vector& v)
	{
		for (int i = 0; i < size; i++)
			data[i] /= v[i];
		return *this;
	}

	vector operator+(const vector& v) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] + v[i];
		return r;
	}

	vector operator-(const vector& v) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] - v[i];
		return r;
	}

	vector operator*(const vector& v) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] * v[i];
		return r;
	}

	vector operator/(const vector& v) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] / v[i];
		return r;
	}


	vector& operator+=(real c)
	{
		for (int i = 0; i < size; i++)
			data[i] += c;
		return *this;
	}

	vector& operator-=(real c)
	{
		for (int i = 0; i < size; i++)
			data[i] -= c;
		return *this;
	}

	vector& operator*=(real c)
	{
		for (int i = 0; i < size; i++)
			data[i] *= c;
		return *this;
	}

	vector& operator/=(real c)
	{
		for (int i = 0; i < size; i++)
			data[i] /= c;
		return *this;
	}

	vector operator+(real c) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] + c;
		return r;
	}

	vector operator-(real c) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] - c;
		return r;
	}

	vector operator*(real c) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] * c;
		return r;
	}

	vector operator/(real c) const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i] / c;
		return r;
	}


	vector operator+() const
	{
		return *this;
	}

	vector operator-() const
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = -data[i];
		return r;
	}


	vector& operator++()
	{
		for (int i = 0; i < size; i++)
			++data[i];
		return *this;
	}

	vector operator++(int)
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i]++;
		return r;
	}

	vector& operator--()
	{
		for (int i = 0; i < size; i++)
			--data[i];
		return *this;
	}

	vector operator--(int)
	{
		vector r;
		for (int i = 0; i < size; i++)
			r[i] = data[i]--;
		return r;
	}

protected:
	real data[size];
};


template< typename scalar, typename real, int size >
vector<real, size> operator+(scalar c, const vector<real, size>& v)
{
	vector<real, size> r;
	for (int i = 0; i < size; i++)
		r[i] = c + v[i];
	return r;
}

template< typename scalar, typename real, int size >
vector<real, size> operator-(scalar c, const vector<real, size>& v)
{
	vector<real, size> r;
	for (int i = 0; i < size; i++)
		r[i] = c - v[i];
	return r;
}

template< typename scalar, typename real, int size >
vector<real, size> operator*(scalar c, const vector<real, size>& v)
{
	vector<real, size> r;
	for (int i = 0; i < size; i++)
		r[i] = c * v[i];
	return r;
}

template< typename scalar, typename real, int size >
vector<real, size> operator/(scalar c, const vector<real, size>& v)
{
	vector<real, size> r;
	for (int i = 0; i < size; i++)
		r[i] = c / v[i];
	return r;
}
