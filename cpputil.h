#pragma once

void boosthello();

template<typename T>
void archive(const T& v, string fn);

template<typename T>
T unarchive(string fn);