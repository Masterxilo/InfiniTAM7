/*

Utilities that cannot be compiled by nvcc, mainly boost stuff

*/

//
//  Copyright (c) 2009-2011 Artyom Beilis (Tonkikh)
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/locale.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <ctime>
using namespace boost::locale;
using namespace std;

void boosthello()
{
    generator gen;
    locale loc = gen("");
    // Create system default locale

    locale::global(loc);
    // Make it system global

    cout.imbue(loc);
    // Set as default locale for output

    cout << format("Today {1,date} at {1,time} we had run our first localization example") % time(0)
        << endl;

    cout << "This is how we show numbers in this locale " << as::number << 103.34 << endl;
    cout << "This is how we show currency in this locale " << as::currency << 103.34 << endl;
    cout << "This is typical date in the locale " << as::date << std::time(0) << endl;
    cout << "This is typical time in the locale " << as::time << std::time(0) << endl;
    cout << "This is upper case " << to_upper("Hello World!") << endl;
    cout << "This is lower case " << to_lower("Hello World!") << endl;
    cout << "This is title case " << to_title("Hello World!") << endl;
    cout << "This is fold case " << fold_case("Hello World!") << endl;

}

// serialize 
template<typename T>
void archive(const T& v, string fn) 
{
    std::ofstream ofs(fn);
    assert(ofs);
    boost::archive::text_oarchive oa(ofs);
    oa & v;
}

template void archive<vector<float>>(const vector<float>& v, string fn);
template void archive<vector<unsigned int>>(const vector<unsigned int>& v, string fn);
template void archive<vector<vector<unsigned int>>>(const vector<vector<unsigned int>>& v, string fn);

// deserialize
template<typename T>
T unarchive(string fn)
{
    std::ifstream ifs(fn);
    assert(ifs);
    boost::archive::text_iarchive ia(ifs);
    T v;
    ia & v;
    return v;
}


template vector<float> unarchive<vector<float>>(string fn);
template vector<unsigned int> unarchive<vector<unsigned int>>(string fn);
template vector<vector<unsigned int>> unarchive<vector<vector<unsigned int>>>(string fn);