DBG_UNREFERENCED_PARAMETER(input); // remove if you use the input
switch (i)
{
case 0:
{
    out(0) = 1.f; // TODO wrong derivative for  out(0) = x(0) - x(1); 
}
break;
case 1:
{
    out(0) = 0.f; // TODO wrong derivative for  out(0) = x(0) - x(1); 
}
break;
}