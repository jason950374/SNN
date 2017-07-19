#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

using namespace std;

template <class T> class Matrix {
public:
	T ** elements;
	int sizeRow;
	int sizeCLM;

	Matrix(int sizeRow, int sizeCLM);
	/* To use this constructor, need "new" the elements
		elements = new T *[sizeRow];
		for (int row = 0; row < sizeRow; row++)
			elements[row] = new T[sizeCLM];
	*/
	Matrix(T ** elements, int sizeRow, int sizeCLM);
	// Copy Constructor, for assignment to prevent pointer (T ** elements) aliasing
	Matrix(const Matrix &M);
	~Matrix();
	Matrix operator*(const Matrix& m2);
	Matrix operator+(const Matrix& m2);
	Matrix& operator=(const Matrix& m2);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//implementation of template need to be in the same file
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> Matrix<T>::Matrix(int sizeRow, int sizeCLM) {
	elements = new T *[sizeRow];
	for (int row = 0; row < sizeRow; row++)
		elements[row] = new T[sizeCLM];
	this->sizeRow = sizeRow;
	this->sizeCLM = sizeCLM;
}

template <class T> Matrix<T>::Matrix(T ** elements, int sizeRow, int sizeCLM) {
	this->elements = elements;
	this->sizeRow = sizeRow;
	this->sizeCLM = sizeCLM
}

template<class T> Matrix<T>::Matrix(const Matrix & M)
{
	sizeRow = M.sizeRow;
	sizeCLM = M.sizeCLM;
	elements = new T *[sizeRow];
	for (int row = 0; row < sizeRow; row++)
		elements[row] = new T[sizeCLM];
	for (int row = 0; row < sizeRow; row++) {
		for (int clm = 0; clm < sizeCLM; clm++) {
			elements[row][clm] = M.elements[row][clm];
		}
	}
}

template<class T> Matrix<T>::~Matrix()
{
	delete elements;
}

template<class T> Matrix<T> Matrix<T>::operator*(const Matrix & m2)
{
	Matrix<T> newM = Matrix<T>(sizeRow, m2.sizeCLM);
	if (sizeCLM != m2.sizeRow) {
		cerr << "matrix size mismatch" << endl; return newM;
	}

	for (int row = 0; row < sizeRow; row++) {
		for (int clm = 0; clm < m2.sizeCLM; clm++) {
			for (int sumCnt = 0; sumCnt < sizeCLM; sumCnt++) {
				newM.elements[row][clm] = 
					newM.elements[row][clm] + (elements[row][sumCnt] * m2.elements[sumCnt][clm]);
			}
		}
	}
	return newM;
}

template<class T> Matrix<T> Matrix<T>::operator+(const Matrix & m2)
{
	Matrix<T> newM = Matrix<T>(sizeRow, m2.sizeCLM);
	if ((sizeRow != m2.sizeRow) || (sizeCLM != m2.sizeCLM)) {
		cerr << "matrix size mismatch" << endl; return newM;
	}

	for (int row = 0; row < sizeRow; row++) {
		for (int clm = 0; clm < sizeCLM; clm++) {
			newM.elements[row][clm] = elements[row][clm] + m2.elements[row][clm];
		}
	}
	return newM;
}


template<class T>  Matrix<T>& Matrix<T>::operator=(const Matrix & m2) {
	sizeRow = m2.sizeRow;
	sizeCLM = m2.sizeCLM;

	for (int row = 0; row < sizeRow; row++)
		delete elements[row];
	delete elements;

	elements = new T *[sizeRow];
	for (int row = 0; row < sizeRow; row++)
		elements[row] = new T[sizeCLM];
	for (int row = 0; row < sizeRow; row++) {
		for (int clm = 0; clm < sizeCLM; clm++) {
			elements[row][clm] = m2.elements[row][clm];
		}
	}
	return *this;
}

#endif
