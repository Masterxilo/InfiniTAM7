
    {
       // out[0] = x(0) - 0.5f;
        out(0) = x(0) - x(1); // penalize difference between z-neighboring values
        //out[1] = x(1);
    }