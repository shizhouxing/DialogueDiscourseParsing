#include <cstdio>
#include <vector>
#include <queue>
 
using namespace std;
 
const int inf = ~0U >> 1;

typedef pair<int, pair<double, int> > edge;
const int maxn = 1000;

int N, M, d[maxn], id[maxn], vis[maxn], fa[maxn], fa_id[maxn];
double fa_weight[maxn];
vector<edge> E[maxn], e[maxn];
queue <int> Q;

void addEdge(int x, int y, double w, int id)
{
    e[x].push_back(make_pair(y, make_pair(w, id)));
    E[y].push_back(make_pair(x, make_pair(w, id)));
}

vector<int> directedMST(int N, vector<int> u, vector<int> v, vector<double> w, double &Cost)
{
    vector<int> Ans;
    Ans.resize(u.size());
    for (auto v:Ans) v = 0;
    for (int i = 1; i <= N; ++i)
    {
        E[i].clear(), e[i].clear();
        d[i] = vis[i] = 0;
    }
    for (int i = 0; i < u.size(); ++i)
        addEdge(u[i], v[i], w[i], i);
    vis[1] = 1; Q.push(1);
    while (Q.size())
    {
        int u = Q.front(); Q.pop();
        for (auto v:e[u])
            if (!vis[v.first])
            {
                vis[v.first] = 1;
                Q.push(v.first);
            }
    }
    for (int i = 1; i <= N; ++i)
        if (!vis[i]) 
        { 
            Cost = 1; 
            return Ans; 
        }
    Cost = 0;
    for (int i = 2; i <= N; ++i)
    {
        fa_weight[i] = inf;
        for (auto v:E[i])
            if (v.second.first < fa_weight[i])
            {
                fa_weight[i] = v.second.first;
                fa_id[i] = v.second.second;
                fa[i] = v.first;
            }
        d[fa[i]]++;
    }
    for (int i = 1; i <= N; ++i) if (!d[i]) Q.push(i);
    while (Q.size())
    {
        int u = Q.front(); Q.pop();
        --d[fa[u]];
        if (!d[fa[u]]) Q.push(fa[u]);
    }
    int nxt_N = 0;
    for (int i = 1; i <= N; ++i) vis[i] = 0;
    for (int i = 1; i <= N; ++i)
        if (!d[i]) 
            id[i] = ++nxt_N;
        else if (!vis[i])
        {
            id[i] = ++nxt_N;
            vis[i] = 1;
            Ans[fa_id[i]] = 1; Cost += fa_weight[i];
            for (int j = fa[i]; j != i; j = fa[j])
                id[j] = nxt_N, vis[j] = 1, Ans[fa_id[j]] = 1, Cost += fa_weight[j];
        }
    if (nxt_N == N) 
    {
        for (int i = 2; i <= N; ++i)
            Ans[fa_id[i]] = 1, Cost += fa_weight[i];
        return Ans;
    }
    vector<int> _u, _v;
    vector<double> _w;
    _u.clear(); _v.clear(); _w.clear();
    vector<int> Plus, Minus;
    Plus.resize(u.size()); Minus.resize(u.size());
    for (int i = 1; i <= N; ++i)
        for (auto v:e[i])
        {
            if (id[i] == id[v.first]) continue;
            if (!d[v.first])
            {
                Plus[_u.size()] = v.second.second;
                Minus[_u.size()] = -1;
                _u.push_back(id[i]);
                _v.push_back(id[v.first]);
                _w.push_back(v.second.first);
            }
            else
            {
                Plus[_u.size()] = v.second.second;
                Minus[_u.size()] = fa_id[v.first];
                _u.push_back(id[i]);
                _v.push_back(id[v.first]);
                _w.push_back(v.second.first - fa_weight[v.first]);                
            }
        }
    double cost = 0;
    vector<int> next = directedMST(nxt_N, _u, _v, _w, cost);
    Cost += cost;
    for (int i = 0; i < next.size(); ++i)
        if (next[i])
        {
            Ans[Plus[i]] = 1;
            if (Minus[i] != -1) Ans[Minus[i]] = 0;
        }
    return Ans;
}

int main()
{
    vector<int> u, v;
    vector<double> w;
    u.clear(); v.clear(); w.clear();
    
    scanf("%d%d", &N, &M);
    for (int i = 1; i <= M; ++i)
    {
        int _u, _v;
        double _w;
        scanf("%d%d%lf", &_u, &_v, &_w);
        u.push_back(_u); v.push_back(_v); w.push_back(_w);
    }
    
    double ans;
    
    vector<int> Ans = directedMST(N, u, v, w, ans);
    printf("%.10lf\n", ans);
    for (int i = 0; i < w.size(); ++i)
        if (Ans[i]) printf("%d ", i);
    puts("");
    return 0;
}
